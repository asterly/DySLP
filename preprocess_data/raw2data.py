import csv
import numpy as np
import json
import time


def get_item():
    itemid2num = {}
    itemnum2id = {}

    cate_ids = []
    cate_level1_ids = []
    brand_ids = []
    shop_ids = []
    item_info_nomal = {}

    with open("../raw/item_info.json") as f:
        item_info = json.load(f)
        for item in item_info:
            item_id = item['item_id']
            if item_id not in itemid2num:
                itemid2num[item_id] = len(itemid2num)
                itemnum2id[str(len(itemnum2id))] = item_id
                cate_ids.append(item['cate_id'])
                cate_level1_ids.append(item['cate_level1_id'])
                brand_ids.append(item['brand_id'])
                shop_ids.append(item['shop_id'])
        f.close()
    normal_cate_ids=normal(cate_ids)
    normal_cate_level1_ids=normal(cate_level1_ids)
    normal_brand_ids=normal(brand_ids)
    normal_shop_ids=normal(shop_ids)

    for index in range(len(itemid2num)) :
        item_id=itemnum2id[str(index)]
        item_info_nomal[item_id]=[normal_cate_ids[index]]
        item_info_nomal[item_id].append(normal_cate_level1_ids[index])
        item_info_nomal[item_id].append(normal_brand_ids[index])
        item_info_nomal[item_id].append(normal_shop_ids[index])
    return item_info_nomal,itemid2num

def get_user_map():
    userid2num = {}
    usernum2id = {}
    with open("../raw/user_info.json") as f:
        user_info = json.load(f)
        for user in user_info:
            user_id = user['user_id']
            if user_id not in userid2num:
                userid2num[user_id] = len(userid2num)
                usernum2id[str(len(usernum2id))] = user_id
        f.close()
    return userid2num , usernum2id

def load_data_single_edge():
    userid2num, usernum2id=get_user_map()
    item_info_nomal,_ = get_item()

    print("-" * 50, end="开始处理单边方案")
    print("-" * 50)
    csv_file = open('../datasets/single_edge_data/single_edge_data.csv', 'w', newline='', encoding='utf8')
    csv_writer = csv.writer(csv_file)
    with open("../raw/item_share_train_info.json") as f :
        json_data=json.load(f)
        json_data.sort(key=lambda x: x["timestamp"])
        idx=0
        for item in json_data:
            idx+=1
            if idx%1000==0:
                print("-"*50,end=str(idx))
                print("-"*50)
            inviter_id=item['inviter_id']
            voter_id=item['voter_id']
            item_id=item['item_id']
            #print("inviter_id_num {} ------ voter_id_num {} ".format(userid2num[inviter_id],userid2num[voter_id]))

            unit_row=[]
            unit_row.append(userid2num[inviter_id])
            unit_row.append(userid2num[voter_id])
            unit_row.append(date_to_timestamp(item['timestamp']))
            unit_row.append(0)
            unit_row=unit_row+item_info_nomal[item_id]
            csv_writer.writerow(unit_row)
        f.close()
    csv_file.close()
    print("-" * 50, end="单边方案处理结束")
    print("-" * 50)


def load_data_double_edge():
    '''
    数据转换方案二，每一条分享记录拆分成图的两条边
    :return:
    '''
    userid2num, usernum2id = get_user_map()
    item_info_nomal,itemid2num = get_item()

    print("-" * 50,end="开始处理双边方案")
    print("-" * 50)
    csv_file = open('../datasets/double_edge_data/double_edge_data.csv', 'w', newline='', encoding='utf8')
    csv_writer = csv.writer(csv_file)
    with open("../raw/item_share_train_info.json") as f:
        json_data = json.load(f)
        json_data.sort(key=lambda x: x["timestamp"])
        idx = 0
        for item in json_data:
            idx += 1
            if idx % 1000 == 0:
                print("-" * 50, end=str(idx))
                print("-" * 50)
            inviter_id = item['inviter_id']
            voter_id = item['voter_id']
            item_id = item['item_id']
            # print("inviter_id_num {} ------ voter_id_num {} ".format(userid2num[inviter_id],userid2num[voter_id]))

            unit_row = []
            unit_row.append(userid2num[inviter_id])
            unit_row.append(itemid2num[item_id])
            unit_row.append(date_to_timestamp(item['timestamp']))
            unit_row.append(1)
            unit_row = unit_row + item_info_nomal[item_id]
            csv_writer.writerow(unit_row)

            unit_row = []
            unit_row.append(userid2num[voter_id])
            unit_row.append(itemid2num[item_id])
            unit_row.append(date_to_timestamp(item['timestamp']))
            unit_row.append(2)
            unit_row = unit_row + item_info_nomal[item_id]
            csv_writer.writerow(unit_row)

        f.close()
    csv_file.close()
    print("-" * 50, end="双边边方案处理结束")
    print("-" * 50)

def normal(arr):

    arr=np.array(arr).astype(np.int32)
    min_val = 1000000
    max_val = 1
    for i in arr:
        if i<min_val and i !=-1 : min_val=i
        if i>max_val : max_val=i

    nomal_arr = []
    for x in arr:
        if x==-1 :
            x=0
        else:
            x = float(x - min_val) / (max_val - min_val)
        nomal_arr.append(x)
    return nomal_arr

#将时间字符串转换为10位时间戳，时间字符串默认为2017-10-01 13:37:04格式
def date_to_timestamp(date, format_string="%Y-%m-%d %H:%M:%S"):
    time_array = time.strptime(date, format_string)
    time_stamp = int(time.mktime(time_array))
    return time_stamp
def main():
   load_data_single_edge()
   load_data_double_edge()

if __name__=="__main__":
    main()