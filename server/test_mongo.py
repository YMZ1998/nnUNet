import pymongo
from pymongo.errors import ConnectionFailure, OperationFailure

url = 'mongodb://datu_super_root:c74c112dc3130e35e9ac88c90d214555__strong@' + '127.0.0.1' + ':27227/default_db?authSource=datu_data&directConnection=true'

try:
    # 建立连接
    client = pymongo.MongoClient(url, tz_aware=True)

    mongo_db = client['datu_data']
    rtss = mongo_db["rtss"].find_one({"ref_image_guid": '1.2.276.0.7230010.3.1.4.577960485.22616.1760491808.379'},
                                     {"_id": 1, "roi_list": 1, "ref_patient_guid": 1})
    print("rtss: ", rtss)

    # 发送一个 ping 来测试连接
    client.admin.command("ping")
    print("MongoDB 连接成功！")

    # 测试获取数据库列表
    print("数据库列表：", client.list_database_names())

    # 测试访问一个数据库和集合
    db = client["test_db"]
    collection = db["test_collection"]

    # 插入测试文档
    result = collection.insert_one({"msg": "hello mongo!"})
    print("插入成功，ID =", result.inserted_id)

    # 查询文档
    doc = collection.find_one()
    print("查询结果：", doc)

except ConnectionFailure:
    print("错误：无法连接到 MongoDB，请检查 URL 或服务器状态。")

except OperationFailure as e:
    print("权限错误：", e)

except Exception as e:
    print("发生异常：", e)
