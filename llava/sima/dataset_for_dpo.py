import argparse
import json
import os


def dump_to_jsonl(obj: list[dict], path: str):
    with open(path, 'w') as file:
        file.writelines([json.dumps(x) + '\n' for x in obj])

def make_dataset(args):

    datas = []
    with open(args.critic_result_path, 'r', encoding='utf-8') as file:
        for line in file:
            datas.append(json.loads(line))

    dpo_datas = []
    for data in datas:
        responses = [data['temperature_decodes_1'], data['greedy_decodes']]
        if data['rank_result'].split('[[')[-1].split(']]')[0] == '1':
            dpo_datas.append({"id": data['id'].split('.')[0],
                              "image": data['id'],
                              "conversations": [
                                  {
                                        "from": "human",
                                        "value": "<image>\n{}".format(data['question'].split('Question: [')[-1].split(']\n')[0])
                                  },
                                  {
                                        "from": "gpt",
                                        "value": responses[0]
                                  }
                              ],
                              "rejected_conversations": [
                                {
                                    "from": "human",
                                    "value": "<image>\n{}".format(data['question'].split('Question: [')[-1].split(']\n')[0])
                                },
                                {
                                    "from": "gpt",
                                    "value": responses[1]
                                }
                            ]})
        elif data['rank_result'].split('[[')[-1].split(']]')[0] == '2':
            dpo_datas.append({"id": data['id'].split('.')[0],
                              "image": data['id'],
                              "conversations": [
                                  {
                                        "from": "human",
                                        "value": "<image>\n{}".format(data['question'].split('Question: [')[-1].split(']\n')[0])
                                  },
                                  {
                                        "from": "gpt",
                                        "value": responses[1]
                                  }
                              ],
                              "rejected_conversations": [
                                {
                                    "from": "human",
                                    "value": "<image>\n{}".format(data['question'].split('Question: [')[-1].split(']\n')[0])
                                },
                                {
                                    "from": "gpt",
                                    "value": responses[0]
                                }
                            ]})

    with open(os.path.join(args.dataset_save_path, args.dataset_save_name), 'w') as f:
        json.dump(dpo_datas, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--critic-result-path", type=str, default=None)
    parser.add_argument("--dataset-save-path", type=str, default=None)
    parser.add_argument("--dataset-save-name", type=str, default=None)
    args = parser.parse_args()

    make_dataset(args)