import argparse
import datetime as dt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
import json
import pandas as pd
pd.set_option('display.max_columns', None)
import requests
import time
from tqdm import tqdm


# Variables
parser = argparse.ArgumentParser()
bool_parser = parser.add_mutually_exclusive_group(required=False)
bool_parser.add_argument('--china', dest='china', action='store_true')
bool_parser.add_argument('--no-china', dest='china', action='store_false')
parser.set_defaults(flag=True)
parser.add_argument('--searchResults_only', type=int, default=0)
parser.add_argument('--taskId', type=str, default='None')
parser.add_argument('--offset', type=int, default=0)
args = parser.parse_args()
china = args.china
searchResults_only = args.searchResults_only
taskId = args.taskId
offset = args.offset
headers = {'Content-Type': 'application/x-www-form-urlencoded'}
grant_type = 'password'
if china == True:
    base_url = 'https://advancedapi.bazhuayu.com/'
    username = 'cathayholdingsct'
    password = 'Jj70827406'
    taskGroup = '498872'
    if taskId == 'None':
        if searchResults_only != 0:
            taskIdList = ['07d814e1-21c9-d902-fa05-c11be09abdea', '8ffb6025-026e-f3a3-31f9-bd2b842f7c00', 'b382f214-a56c-9e7c-9d01-9258228de53d']
        else:
            taskIdList = ['07d814e1-21c9-d902-fa05-c11be09abdea', '8ffb6025-026e-f3a3-31f9-bd2b842f7c00',
                          'b382f214-a56c-9e7c-9d01-9258228de53d', '35e517d8-79f5-5b30-c5bb-691c2bc03560',
                          'e167ca9f-42d1-e696-9eb1-ffec3db33be3', 'f5b3329a-6969-5c0a-d7e8-6dd90b93fe21',
                          '94af7f13-536f-79e3-2331-b6f611583c14', '2d7a6a56-1710-4b79-20d9-86efbeb4e072']
    else:
        taskIdList = [taskId]
else:
    base_url = 'https://advancedapi.octoparse.com/'
    username = 'cathaylab'
    password = 'Mintyui@99'
    taskGroup = '498872'
    if taskId == 'None':
        if searchResults_only != 0:
            taskIdList = ['94a34d30-0339-966e-7c09-0ac0fcf1fad4', 'be3552d4-bc20-2780-f136-5857e84aab36',
                          '69a9abf8-4efe-5927-2bbe-23115de446e3', 'e5843d8e-ef4c-3298-8912-80efd920be4c',
                          '88c8e9b6-c2ec-0612-1c9c-02392d915f04', 'ff50becc-a7d8-1d2c-cfdd-879ef4904d56',
                          '44527806-84fc-a859-71fc-0b86495689c1'   ]
        else:
            taskIdList = ['94a34d30-0339-966e-7c09-0ac0fcf1fad4', '69a9abf8-4efe-5927-2bbe-23115de446e3',
                          '43e1698c-6093-4c5d-20a2-cf3d03f57874', 'be3552d4-bc20-2780-f136-5857e84aab36',
                          'cadb30fd-9899-4ed6-94fa-d52765b0e4d1', '6f7054c6-cc9f-670c-7db9-5d5c5b53513a',
                          '88c8e9b6-c2ec-0612-1c9c-02392d915f04', 'ff50becc-a7d8-1d2c-cfdd-879ef4904d56',
                          '1cc12c4e-f5f1-45f0-f709-fd6050b7ea80', '44527806-84fc-a859-71fc-0b86495689c1',
                          'e5843d8e-ef4c-3298-8912-80efd920be4c', 'abb7a4ff-32c3-0bbd-d21e-b3d64e656756',
                          'df903738-8adc-72e4-d301-4e016ee7ff1f', 'df99c1cb-fc26-6743-8775-60502f124183']
    else:
        taskIdList = [taskId]


# Function
def getAccessToken(base_url, username, password):
    r = requests.post(base_url + 'token',
                      data={'username': username,
                            'password': password,
                            'grant_type': 'password'})
    print(r.status_code, r.reason)
    print(json.dumps(json.loads(r.text), indent=4))
    auth = json.loads(r.text)
    access_token = auth['access_token']
    token_type = auth['token_type']
    refresh_token = auth['refresh_token']
    api_headers = {'Authorization': f'{token_type} {access_token}'}

    return api_headers, refresh_token


def getData(base_url, api_headers, taskId, offset=0):
    cate = 'task/GetTaskStatusByIdList'
    api_url = base_url + 'api/' + cate
    r = requests.post(api_url, headers=api_headers, data={'taskIdList': [taskId]})
    task_name = json.loads(r.text)['data'][0]['taskName']
    task_name = task_name.replace('|', ',')
    print(task_name)

    cate = 'alldata/GetDataOfTaskByOffset'
    api_url = base_url + 'api/' + cate
    r = requests.get(api_url, headers=api_headers, params={'taskId': taskId, 'offset': 0, 'size': 1})
    total_row = json.loads(r.text)['data']['total']
    restofrows = json.loads(r.text)['data']['restTotal']
    timestamp = dt.datetime.now().strftime('%Y%m%dT%H%M%S')
    print('total_row:', total_row)

    if total_row > 0:

        offset = offset
        size = 1000
        export_data = pd.DataFrame(json.loads(r.text)['data']['dataList'])

        pbar = tqdm(total=int(round(total_row / size, 0)) + 1)
        while restofrows:
            try:
                r = requests.get(api_url, headers=api_headers, params={'taskId': taskId, 'offset': offset, 'size': size})
            except:
                break

            restofrows = json.loads(r.text)['data']['restTotal']

            if len(export_data) == 1:
                export_data = pd.DataFrame(json.loads(r.text)['data']['dataList'])
            else:
                export_data = pd.concat([export_data, pd.DataFrame(json.loads(r.text)['data']['dataList'])])
                if (offset % 100000) == 0:  # 每 10 萬筆存一次
                    print(export_data.shape)
                    print(export_data.head())
                    export_data.to_csv(f'octoparse_export/{task_name}-{timestamp}-{offset}.csv', index=False)

            offset += 1000
            pbar.update(1)
            time.sleep(1)

        pbar.close()

        if export_data.shape[0] == total_row:
            print('all rows exported.')
        else:
            print(f'Total rows: {total_row}, only exported: {export_data.shape}')

    else:
        export_data = 0

    return task_name, export_data, offset


# Main
print(base_url)
print('\n Export searchResults only\n'  if searchResults_only != 0 else 'Export all')
api_headers, refresh_token = getAccessToken(base_url, username, password)

for _id in range(0, len(taskIdList)):

    print(_id)
    try:
        task_name, export_data, offset = getData(base_url=base_url, api_headers=api_headers, taskId=taskIdList[_id], offset=offset)
    except:
        task_name, export_data, offset = getData(base_url=base_url, api_headers=api_headers, taskId=taskIdList[_id], offset=offset)

    offset = 0
    timestamp = dt.datetime.now().strftime('%Y%m%dT%H%M%S')
    if type(export_data) != int:
        print(export_data.shape)
        print(export_data.head())
        export_data.to_csv(f'octoparse_export/{task_name}-{timestamp}.csv', index=False)
        print(f'{task_name}-{timestamp}.csv has been saved successfully.')
    else:
        print(f'{task_name} has no data to export.')


