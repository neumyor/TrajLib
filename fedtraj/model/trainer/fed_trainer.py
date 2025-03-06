import time
import logging
import pickle
import torch
import torch.nn as nn
import heapq
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial

from fedtraj.utils.data_loader import read_traj_dataset
from fedtraj.utils.traj import *
from fedtraj.utils import tool_funcs

from .utils import collate_and_augment, collate_for_test


class BaseClient:
    def __init__(self, id, str_aug1, str_aug2, dataset, model_class):
        self.batch_size = Config.trajcl_batch_size
        self.trainset = dataset
        self.testset = None
        self.id = id

        self.E = Config.E
        self.aug1 = get_aug_fn(str_aug1)
        self.aug2 = get_aug_fn(str_aug2)

        self.embs = (
            pickle.load(open(Config.dataset_embs_file, "rb")).to("cpu").detach()
        )  # tensor
        self.cellspace = pickle.load(open(Config.dataset_cell_file, "rb"))
        self.temperature = 0.07

        # if self.trainset != None:
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=Config.trajcl_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            collate_fn=partial(
                collate_and_augment,
                cellspace=self.cellspace,
                embs=self.embs,
                augfn1=self.aug1,
                augfn2=self.aug2,
            ),
        )

        # if self.testset != None:
        #     self.testloader = DataLoader(
        #         self.testset,
        #         batch_size=Config.trajcl_batch_size,
        #         shuffle=False,
        #         num_workers=0,
        #         drop_last=True,
        #         collate_fn=partial(collate_and_augment, cellspace=self.cellspace,
        #                            embs=self.embs, augfn1=self.aug1, augfn2=self.aug2)
        #     )

        self.E = 1
        self.device = Config.device
        self.model = model_class().to(Config.device)
        self.last_model = model_class().to(Config.device)
        self.global_model = model_class().to(Config.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=Config.trajcl_training_lr, weight_decay=0.0001
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=Config.trajcl_training_lr_degrade_step,
            gamma=Config.trajcl_training_lr_degrade_gamma,
        )

    def local_train(self):
        raise NotImplementedError()

    def clone_model(self, target):
        p_tensor = target.model.parameters_to_tensor()
        self.model.tensor_to_parameters(p_tensor)
        return


class BaseServer(BaseClient):
    def __init__(self, id, str_aug1, str_aug2, dataset, model_class):
        super().__init__(id, str_aug1, str_aug2, dataset, model_class)
        self.n_clients = Config.cls_num
        self.n_clients_per_round = round(self.n_clients * Config.C)
        self.learning_rate = Config.trajcl_training_lr
        self.step = 0
        self.clients = []

    def aggregate_model(self):
        raise NotImplementedError()

    def train(self):
        # finish 1 comm round
        raise NotImplementedError()

    def sample_client(self):
        return random.sample(
            self.clients,
            self.n_clients_per_round,
        )


class Client(BaseClient):
    def local_train(self):
        param = self.model.parameters_to_tensor()
        self.global_model.tensor_to_parameters(param)
        loss_ep_avg = 0
        for epoch in range(self.E):
            _time_ep = time.time()
            loss_ep = []
            train_gpu = []
            train_ram = []

            _time_batch_start = time.time()
            for i_batch, batch in enumerate(self.trainloader):
                _time_batch = time.time()
                self.optimizer.zero_grad()

                (
                    trajs1_emb,
                    trajs1_emb_p,
                    trajs1_len,
                    trajs2_emb,
                    trajs2_emb_p,
                    trajs2_len,
                ) = batch

                model_rtn = self.model(
                    trajs1_emb,
                    trajs1_emb_p,
                    trajs1_len,
                    trajs2_emb,
                    trajs2_emb_p,
                    trajs2_len,
                )
                loss = self.model.loss(*model_rtn)

                loss *= np.random.randn() / 100 + 1

                loss.backward()
                self.optimizer.step()
                loss_ep.append(loss.item())
                train_gpu.append(tool_funcs.GPUInfo.mem()[0])
                train_ram.append(tool_funcs.RAMInfo.mem())

                if i_batch % 100 == 0 and i_batch:
                    logging.debug(
                        "[Training] client{} ep-batch={}-{}, loss={:.3f}, @={:.3f}, gpu={}, ram={}".format(
                            self.id,
                            epoch,
                            i_batch,
                            loss.item(),
                            time.time() - _time_batch_start,
                            tool_funcs.GPUInfo.mem(),
                            tool_funcs.RAMInfo.mem(),
                        )
                    )

            # self.scheduler.step()

            loss_ep_avg = tool_funcs.mean(loss_ep)
            logging.info(
                "[Training] client{} ep={}: avg_loss={:.3f}, gpu={}, ram={}".format(
                    self.id,
                    epoch,
                    loss_ep_avg,
                    tool_funcs.GPUInfo.mem(),
                    tool_funcs.RAMInfo.mem(),
                )
            )

            training_gpu_usage = tool_funcs.mean(train_gpu)
            training_ram_usage = tool_funcs.mean(train_ram)
        param = self.model.parameters_to_tensor()
        self.last_model.tensor_to_parameters(param)
        return loss_ep_avg


class Server(BaseServer):
    def aggregate_model(self, clients):
        n = len(clients)
        p_tensors = []
        for _, client in enumerate(clients):
            p_tensors.append(client.model.parameters_to_tensor())
        avg_tensor = sum(p_tensors) / n
        self.model.tensor_to_parameters(avg_tensor)

    def train(self):
        # random clients
        clients = self.sample_client()

        for client in clients:
            # send params
            client.clone_model(self)
            for p in client.optimizer.param_groups:
                p["lr"] = self.learning_rate

        loss_ep_sum = 0.0
        for client in clients:
            # local train
            loss_ep_sum += client.local_train()

        # aggregate params
        self.aggregate_model(clients)

        self.step += 1

        if self.step % Config.trajcl_training_lr_degrade_step == 0:
            self.learning_rate *= Config.trajcl_training_lr_degrade_gamma

        return clients, loss_ep_sum / self.n_clients_per_round, 0


class FedTrajCLTrainer:

    def __init__(self, str_aug1, str_aug2, model_class):

        # import ourmethod.fed_trajcl as Fed
        super(FedTrajCLTrainer, self).__init__()

        self.aug1 = get_aug_fn(str_aug1)
        self.aug2 = get_aug_fn(str_aug2)

        self.embs = (
            pickle.load(open(Config.dataset_embs_file, "rb")).to("cpu").detach()
        )  # tensor
        # self.embs, _ = torch.sort(self.embs)
        self.cellspace = pickle.load(open(Config.dataset_cell_file, "rb"))

        train_dataset, _, test_dataset = read_traj_dataset(Config.dataset_file)
        # full length trajectory
        # if Config.ldp == 1:
        #     train_dataset_split = tool_funcs.split(
        #         get_ldp_dataset(train_dataset), Config.cls_num
        #     )
        # else:
        #     train_dataset_split = tool_funcs.split(train_dataset, Config.cls_num)

        if Config.ldp == 1:
            train_dataset = get_ldp_dataset(train_dataset)

        from fedtraj.model.trainer.utils import get_federated_segments, cut_trajectorys_into_segments
        from fedtraj.utils.data_loader import TrajDataset
        train_dataset_split = get_federated_segments(train_dataset, Config.cls_num)
        test_dataset = TrajDataset(cut_trajectorys_into_segments(test_dataset.data))

        # init clients
        self.clients = []
        for i in range(Config.cls_num):
            id = i + 1
            client = Client(id, str_aug1, str_aug2, train_dataset_split[i], model_class)
            self.clients.append(client)
        # init server
        self.server = Server(0, str_aug1, str_aug2, test_dataset, model_class)
        self.server.clients = self.clients

        self.checkpoint_file = "{}/{}_TrajCL_best_{}.pt".format(
            Config.checkpoint_dir, Config.dataset_prefix, Config.dumpfile_uniqueid
        )

    def train(self):
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        logging.info("[Training] START! timestamp={:.0f}".format(training_starttime))
        torch.autograd.set_detect_anomaly(True)

        best_loss_train = 100000
        best_epoch = 0
        bad_counter = 0
        bad_patience = Config.trajcl_training_bad_patience

        for i_ep in range(Config.trajcl_training_epochs):
            _time_ep = time.time()

            Config.fedproc_loss_weight = (
                1 - (i_ep + 1) / Config.trajcl_training_epochs / 10
            )

            _, loss_ep_avg, delta = self.server.train()

            logging.info(
                "[Training] ep={}: avg_loss={:.3f}, delta={:.3f}, @={:.3f}/{:.3f}, time={}".format(
                    i_ep,
                    loss_ep_avg,
                    delta,
                    time.time() - _time_ep,
                    time.time() - training_starttime,
                    time.time() - training_starttime,
                )
            )

            # early stopping
            if abs(loss_ep_avg - best_loss_train) > 1e-2:
                best_epoch = i_ep
                best_loss_train = loss_ep_avg
                bad_counter = 0
                self.save_checkpoint()
            else:
                bad_counter += 1

            if (
                bad_counter == bad_patience
                or (i_ep + 1) == Config.trajcl_training_epochs
            ):
                logging.info(
                    "[Training] END! @={}, best_epoch={}, best_loss_train={:.6f}".format(
                        time.time() - training_starttime, best_epoch, best_loss_train
                    )
                )
                break

        return {
            "enc_train_time": time.time() - training_starttime,
            "enc_train_gpu": training_gpu_usage,
            "enc_train_ram": training_ram_usage,
        }

    @torch.no_grad()
    def test(self):
        # 1. read best model
        # 2. read trajs from file, then -> embeddings
        # 3. run testing
        # n. varying downsampling rates, and distort rates

        logging.info("[Test]start.")
        self.load_checkpoint()
        self.server.model.eval()

        # varying downsampling; varying distort
        vt = Config.test_type
        results = []
        for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
            with open(
                Config.dataset_file + "_newsimi_" + vt + "_" + str(rate) + ".pkl", "rb"
            ) as fh:
                q_lst, db_lst = pickle.load(fh)
                querys, databases = self.test_merc_seq_to_embs(q_lst, db_lst)
                dists = torch.cdist(querys, databases, p=1)  # [1000, 100000]
                targets = torch.diag(dists)  # [1000]
                result = torch.sum(torch.le(dists.T, targets)).item() / len(q_lst)
                results.append(result)
        logging.info(
            "[EXPFlag]task=newsimi,encoder=TrajCL,varying={},r1={:.3f},r2={:.3f},r3={:.3f},r4={:.3f},r5={:.3f}".format(
                vt, *results
            )
        )
        return

    @torch.no_grad()
    def test_merc_seq_to_embs(self, q_lst, db_lst):
        querys = []
        databases = []
        num_query = len(q_lst)  # 1000
        num_database = len(db_lst)  # 100000
        batch_size = num_query

        for i in range(num_database // batch_size):
            if i == 0:
                trajs1_emb, trajs1_emb_p, trajs1_len = collate_for_test(
                    q_lst, self.cellspace, self.embs
                )
                trajs1_emb = self.server.model.interpret(
                    trajs1_emb, trajs1_emb_p, trajs1_len
                )
                querys.append(trajs1_emb)

            trajs2_emb, trajs2_emb_p, trajs2_len = collate_for_test(
                db_lst[i * batch_size : (i + 1) * batch_size], self.cellspace, self.embs
            )
            trajs2_emb = self.server.model.interpret(
                trajs2_emb, trajs2_emb_p, trajs2_len
            )
            databases.append(trajs2_emb)

        querys = torch.cat(querys)  # tensor; traj embeddings
        databases = torch.cat(databases)
        return querys, databases

    def dump_embeddings(self):
        return

    def save_checkpoint(self):
        torch.save(
            {
                "model_state_dict": self.server.model.state_dict(),
                "aug1": self.aug1.__name__,
                "aug2": self.aug2.__name__,
            },
            self.checkpoint_file,
        )
        return

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_file, weights_only=True)
        self.server.model.load_state_dict(checkpoint["model_state_dict"])
        self.server.model.to(Config.device)

        return

    def knn_test(self, fn_name="edwp"):
        # 1. Read beat model
        # 2. Get 1000 querys and 10000 database
        # 3. Collate for knn test
        # 4. Calculate tradition knn using traj_dist
        # 5. count hit ratio and recall ratio

        # 1
        logging.info("[Knn-Test]start.")
        start_time = time.time()
        self.load_checkpoint()
        self.server.model.eval()
        fn = tool_funcs.get_simi_fn(fn_name)

        # 2
        with open(Config.dataset_file + "_newsimi_raw.pkl", "rb") as fh:
            _, db_lst = pickle.load(fh)
            q_lst, db_lst = db_lst[1000:2000], db_lst[2000:12000]

            # 3
            querys, databases = self.test_merc_seq_to_embs(q_lst, db_lst)
            ret_lstses = tool_funcs.normalization([q_lst, db_lst])
            normal_q, normal_db = ret_lstses[0], ret_lstses[1]
            # normal_q, normal_db = q_lst, db_lst
            points = [5, 10, 20, 50]
            hr = [[] for _ in range(4)]
            for i, query in enumerate(querys):
                normal_i = normal_q[i]
                print(i, time.time() - start_time)

                heap_b = []
                for j, database in enumerate(databases):
                    dist = -torch.norm(query - database).item()
                    if len(heap_b) < 50 or dist > heap_b[0][0]:
                        heapq.heappush(heap_b, (dist, j))
                    if len(heap_b) > 50:
                        heapq.heappop(heap_b)
                heap_b = sorted(heap_b, reverse=True)

                heap_a = []
                for j, normal_j in enumerate(normal_db):
                    simi = -fn(normal_i, normal_j)
                    if len(heap_a) < 50 or simi > heap_a[0][0]:
                        heapq.heappush(heap_a, (simi, j))
                    if len(heap_a) > 50:
                        heapq.heappop(heap_a)
                heap_a = sorted(heap_a, reverse=True)

                if i == 0:
                    for value, id in heap_b:
                        print(value, id)
                    print("=======================")
                    for simi, id in heap_a:
                        print(simi, id)

                for idx, point in enumerate(points):
                    cnt = 0
                    for now_i in range(point):
                        for now_j in range(point):
                            if heap_b[now_i][1] == heap_a[now_j][1]:
                                cnt += 1
                    hr[idx].append(cnt)
                    print(i, point, cnt)

        for idx in range(4):
            cnt = 0
            for num in hr[idx]:
                cnt += num
            print(cnt / points[idx])

        return
