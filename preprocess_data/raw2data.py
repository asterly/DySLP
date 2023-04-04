import csv
import numpy as np
import json
import time

def load_json_file():
    userid2num = {}
    usernum2id = {}
    itemid2num = {}
    itemnum2id = {}
    cate_ids=[]
    cate_level1_ids=[]
    brand_ids=[]
    shop_ids=[]
    item_info_nomal={}
    with open("../raw/user_info.json") as f :
        user_info=json.load(f)
        for user in user_info:
            user_id = user['user_id']
            if user_id not in userid2num:
                userid2num[user_id] = len(userid2num)
                usernum2id[str(len(usernum2id))] = user_id
        f.close()

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
    print("-"*120)
    csv_file = open('/home/akun/data_20230404.csv', 'w', newline='', encoding='utf8')
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
   load_json_file()

if __name__=="__main__":
    main()