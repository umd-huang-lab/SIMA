import json
import argparse
import os

def dump_to_jsonl(obj: list[dict], path: str):
    with open(path, 'w') as file:
        file.writelines([json.dumps(x) + '\n' for x in obj])


def make_prompts(args):

    datas = []
    with open(args.self_response_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                datas.append(json.loads(line))
            except Exception as e:
                print(f"An error occurred: {e}. Skipping this batch.")
                continue


    prompts = []
    for data in datas:
        prompt = "You are provided with a question about the image, a ground truth and two AI generated responses based on the image. \
        Please review two AI generated responses carefully and compare the AI generated responses with the ground truth. \
        You need to choose one AI generated response better aligned with the ground truth. \
        You should consider the following factors: \
        1. Accuracy in Object Description: Evaluate the accuracy of the descriptions concerning the objects mentioned in the ground truth answer. Responses should minimize the mention of objects not present in the ground truth answer, and inaccuracies in the description of existing objects. \
        2. Accuracy in Depicting Relationships: Consider how accurately the relationships between objects are described compared to the ground truth answer. Rank higher the responses that least misrepresent these relationships. \
        3. Accuracy in Describing Attributes: Assess the accuracy in the depiction of objects' attributes compared to the ground truth answer. Responses should avoid inaccuracies in describing the characteristics of the objects present. \
        Example 1:\n \
        Question: [What could be a possible reason for the man wearing a number one on his chest while riding the horse?]\n\
        Ground truth: [While riding the chestnut-colored horse in the competition, such as a polo match or equestrian event, the man is wearing a number one on his chest to identify his team or individual position. In these types of events, each participant or team member is assigned a unique number for easy identification and tracking during the competition. This allows organizers and spectators to differentiate between participants and teams, helping them follow the progress and performance of each individual or team member throughout the event.]\n\
        AI generated response 1: [A possible reason for the man wearing a number one on his chest while riding the horse could be that he is participating in a competition or event. In such events, riders often wear numbers to identify themselves and their horses. The number one on his chest might indicate that he is the first rider in the competition or event, or it could simply be a unique identifier assigned to him by the organizers.]\n\
        AI generated response 2: [A possible reason for the man wearing a number one on his chest while riding the horse could be that he is participating in a competitive event, such as a horse show, equestrian competition, or even a casual game with friends. In such events, participants often wear numbers to identify themselves and keep track of their performance, placing in the competition, or personal achievements. The number one on his chest could indicate that he is the top-ranked participant, or he is simply using it to distinguish himself from the other competitors.]\n\
        ASSISTANT:\n \
        The reason: AI generated response 1 is more aligned with the ground truth as it strictly suggests participation in a competition or event and does not add unrelated possibilities. Both responses mention the use of numbers for identification, which aligns with the ground truth. AI generated response 2 includes additional speculation about the man being the top-ranked participant or using the number to distinguish himself, which is not mentioned in the ground truth. It also suggests that the event could be a casual game with friends, which is not supported by the ground truth focusing on a competitive context.  \
        The better AI generated response: [[1]].\n\
        Example 2:\n \
        Question: [Analyze the image in a comprehensive and detailed manner.]\n\
        Ground truth: [The image depicts a large horse standing in a field, accompanied by two main individuals. One person, dressed in English-style riding pants and boots, is standing close to the horse, while the other is further back to the right. There are several other people in the backdrop, making the scene appear lively and social. A potted plant can also be found towards the right side of the image, adding a touch of greenery to the scene.]\n\
        AI generated response 1: [The image features a woman standing next to a brown horse in a grassy field. The woman appears to be engaged in a conversation with the horse, possibly discussing its training or care. There are several other people in the background, some of whom might be observing the interaction between the woman and the horse. In addition to the people and the horse, there are two potted plants visible in the scene, one located near the left edge of the image and the other towards the right side. A book can also be seen on the ground, possibly belonging to one of the people present in the field.]\n\
        AI generated response 2: [The image features a woman standing next to a brown horse in a grassy field. The woman appears to be engaged in a conversation with the horse, possibly discussing its training or care. The horse is wearing a bridle, indicating that it is well-trained and responsive to commands. In the background, there are several other people present, possibly observing the interaction between the woman and the horse or engaging in their own activities. The scene captures a moment of connection between the woman and the horse, showcasing the bond between humans and animals.]\n\
        ASSISTANT:\n \
        The reason: AI generated response 1 incorrectly mentions two potted plants and a book on the ground, none of which are described in the ground truth. There's only one potted plant mentioned in the ground truth, and no book is noted. AI generated response 2 does not mention the potted plants or a book, thus avoiding the inaccuracies present in response 1. \
        The better AI generated response: [[2]].\n\
        Your task: \n\
        Question: [{}]\n\
        Ground truth: [{}]\n\
        AI generated response 1: [{}]\n\
        AI generated response 2: [{}]\n\
        ASSISTANT:\n ".format(data['text'].replace("\n", ""),
                             data['ground_truth'].replace("\n", ""),
                             data['temperature_decodes_1'].replace("\n", ""),
                             data['greedy_decodes'].replace("\n", ""))

        prompts.append({'text': prompt,
                          'question': data['text'],
                          'image': data['image_path'],
                          'ground_truth': data['ground_truth'],
                          'temperature_decodes_1': data['temperature_decodes_1'],
                          'greedy_decodes': data['greedy_decodes']})

    dump_to_jsonl(prompts, os.path.join(args.prompt_save_path, args.prompt_save_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-response-path", type=str, default=None)
    parser.add_argument("--prompt-save-path", type=str, default=None)
    parser.add_argument("--prompt-save-name", type=str, default=None)
    args = parser.parse_args()

    make_prompts(args)
