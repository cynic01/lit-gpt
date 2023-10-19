import pandas as pd
from datasets import load_dataset
import pickle

def save_data(filename, data):
    #Storing data with labels
    a_file = open(filename, "wb")
    pickle.dump(data, a_file)
    a_file.close()

def depth_first_traversal(group, parent_id, path=[], paths=[]):
    row_index = group.index[group['message_id'] == parent_id]
    if len(row_index) != 1:
        print('Duplicate entries with same message ID. Traversal not possible. Exiting...')
        exit()

    path.append(row_index[0])

    # Check if the current node is a leaf node (no children)
    row_indices = group.index[group['parent_id'] == parent_id]
    if len(row_indices) == 0:
        # This is a leaf node. Save the current path when reaching a leaf node
        paths.append(path.copy())

    else:
        # Traverse through the children. Find the child nodes of the current node
        # Perform depth-first traversal for each child node
        for index in row_indices:
            child_id = group.loc[index]['message_id']
            path, paths = depth_first_traversal(group, child_id, path, paths)

    # Remove the current node from the path before backtracking
    path.pop()
    
    return path, paths

    


def convert_group_to_conversation(group):
    group.to_csv('preview.csv')
    all_conversations = []

    #find parent_id
    parent_id = None
    for _, row in group.iterrows():
        message_id = row['message_id']
        tree_id = row['message_tree_id']
        lang = row['lang']

        if lang != 'en':
            return all_conversations

        if message_id == tree_id:#this is parent
            parent_id = tree_id
            break

    if parent_id:
        _, all_paths = depth_first_traversal(group, parent_id, [], [])
        for path in all_paths:
            conversation = []
            for index in path:
                conversation.append(group.loc[index])

            all_conversations.append(conversation)
    
    return all_conversations



    
    




if __name__ == '__main__':

    ds = load_dataset("OpenAssistant/oasst1")
    train = ds['train']      # len(train)=84437 (95%)
    val = ds['validation']   # len(val)=4401 (5%)
    print(len(train), len(val))


    for split in ['train', 'validation']:
        dataset = ds[split].to_pandas()
        grouped = dataset.groupby('message_tree_id')

        conversation_data = []
        total_conversations = 0
        total_groups = len(grouped.groups.items())
        for i, (name, group_indices) in enumerate(grouped.groups.items()):
            group_rows = dataset.loc[group_indices]

            temp_conv_data = convert_group_to_conversation(group_rows)
            conversation_data += temp_conv_data
            total_conversations += len(temp_conv_data)

            print(i, 'out of', total_groups, 'TOTAL CONVERSATIONS:', total_conversations)

            
        save_data('oasst1_' + split + '.pkl', conversation_data)


            


