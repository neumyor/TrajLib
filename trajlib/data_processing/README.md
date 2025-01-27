核心功能：将各式各样的轨迹数据集进行预处理，成为统一的数据格式

DataReader：读取文件中的轨迹数据，转换为内部支持的数据类型
DataConverter：在内部支持的数据类型之间进行转换
SpatialMatcher: 将轨迹点匹配到路网或网格的ID上

# 数据的基本格式

**TrajectoryData:**
包含两张具体的表
第一张表是轨迹点表，其中每一行对应一个轨迹点：
- point_id: 轨迹点的独特id，每个轨迹点仅对应一个具体id，为主键
- traj_id: 轨迹的独特id，每条轨迹仅对应一个具体的id
- timestamp：该轨迹点对应的具体时间，采用纯整数
- lon: 经度，可选
- lat：纬度，可选
- road_id：路网上的id，可选
- grid_id：网格上的id，可选
- poi_id：如果是poi轨迹，那么该属性为对应的poi的唯一标识id，可选

第二张表是轨迹表，其中每一行对应一条轨迹
- traj_id：轨迹的独特id
- user_id：属于哪个用户，可选
- travel_time：总旅行时间，可选

**GeoData：**
第一张表是信息表：
- geo_id：地理对象的id
- type：可选geojson中的 [point, linestring, polygon] 三种
  - 对于路口或传感器位置，我们可以采用point来表示
  - 对于道路路段，我们可以采用linestring来表示
  - 对于网格或poi区域，我们可以采用polygon来表示
- coord：表示地理对象的地理位置信息。其与geojson的表示相一致。
  - 对于point，其表示为数组[lon, lat]
  - 对于linestring，其表示为[[lon_1, lat_1]. [lon_2, lat_2],..., [lon_n, lat_n]]
  - 对于polygon，其表示为[[[lon_1, lat_1]. [lon_2, lat_2],..., [lon_n, lat_n]]]。注意比linestring多一层嵌套。
- row_id：若当前表代表网格数据，该列表示在网格中的具体列序号，可选
- column_id：对于网格数据，该列表示在网格中的具体行序号，可选

**GeoRelationData:**
表示了不同地理对象之间的关系
- rel_id：关系表的唯一id
- origin_id：地理对象的id，对应对象的geo_id
- dest_id：地理对象的id，对应对象的geo_id

