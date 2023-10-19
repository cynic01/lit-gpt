import pickle
import numpy as np

def load_data(filename):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output

def get_conversation_text(filename, lang_tag = 'en'):
    data = load_data(filename)

    conversation_dataset_text = []
    for conversation in data:

        if conversation[0]['lang'] == lang_tag:
            conversation_data = []

            for turn in conversation:
                conversation_data.append((turn['role'], turn['text']))


            conversation_dataset_text.append(conversation_data)

    return conversation_dataset_text


if __name__ == '__main__':

    for split in ['train', 'validation']:
        filename = 'oasst1_' + split + '.pkl'
        dataset = load_data(filename)

        conversation_lengths = []
        for conversation in dataset:
            conversation_lengths.append(len(conversation))

        conversation_lengths = np.array(conversation_lengths)
        print(split, ':', np.mean(conversation_lengths), np.std(conversation_lengths), np.max(conversation_lengths), np.min(conversation_lengths))
    

    #conversation_text = get_conversation_text('oasst1_train.pkl')
    #print(len(conversation_text))

