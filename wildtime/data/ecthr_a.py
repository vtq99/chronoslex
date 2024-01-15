import os
import pickle
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .utils import initialize_distilbert_transform, download_detection

PREPROCESSED_FILE = 'ecthr_a.pkl'
MAX_TOKEN_LENGTH = 128
RAW_DATA_FILE = ['ecthr-train.jsonl', 'ecthr-dev.jsonl', 'ecthr-test.jsonl']
ID_HELD_OUT = 0.2
GROUP = 1

class ECtHRABase(Dataset):
    def __init__(self, args):
        super().__init__()

        if args.reduced_train_prop is None:
            self.data_file = f'{str(self)}.pkl'
        else:
            self.data_file = f'{str(self)}_{args.reduced_train_prop}.pkl'
        download_detection(args.data_dir, self.data_file)
        preprocess(args)

        self.datasets = pickle.load(open(os.path.join(args.data_dir, self.data_file), 'rb'))

        self.args = args
        self.ENV = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
        # self.ENV = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
        self.num_classes = 14 # 15 if we remove classes
        self.num_tasks = len(self.ENV)
        self.current_time = 0
        self.mini_batch_size = args.mini_batch_size
        self.task_indices = {}
        self.transform = initialize_distilbert_transform(max_token_length=MAX_TOKEN_LENGTH)
        self.mode = 0

        self.class_id_list = {i: {} for i in range(self.num_classes)}
        start_idx = 0
        self.task_idxs = {}
        self.input_dim = []
        cumulative_batch_size = 0
        for i, year in enumerate(self.ENV):
            # Store task indices
            end_idx = start_idx + len(self.datasets[year][self.mode]['labels'])
            self.task_idxs[year] = [start_idx, end_idx]
            start_idx = end_idx

            # Store class id list
            for classid in range(self.num_classes):
                sel_idx = np.nonzero(np.array(self.datasets[year][self.mode]['labels']) == classid)[0]
                self.class_id_list[classid][year] = sel_idx
            print(f'Year {str(year)} loaded')

            # Store input dim
            num_examples = len(self.datasets[year][self.mode]['labels'])
            cumulative_batch_size += min(self.mini_batch_size, num_examples)
            if args.method in ['erm']:
                self.input_dim.append(cumulative_batch_size)
            else:
                self.input_dim.append(min(self.mini_batch_size, num_examples))

        # total_samples = 0
        # for i in self.ENV:
        #     total_samples += len(self.datasets[i][2]['category'])
        # print('total', total_samples)

    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.datasets[time][self.mode]['text'] = np.concatenate(
            (self.datasets[time][self.mode]['text'], self.datasets[prev_time][self.mode]['text']), axis=0)
        self.datasets[time][self.mode]['labels'] = np.concatenate(
            (self.datasets[time][self.mode]['labels'], self.datasets[prev_time][self.mode]['labels']), axis=0)
        if data_del:
            del self.datasets[prev_time]
        for classid in range(self.num_classes):
            sel_idx = np.nonzero(self.datasets[time][self.mode]['labels'] == classid)[0]
            self.class_id_list[classid][time] = sel_idx

    def update_historical_K(self, idx, K):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.window_start = self.ENV[max(0, idx - K)]
        if idx >= K:
            last_K_num_samples = self.input_dim[idx - K]
            self.datasets[time][self.mode]['text'] = np.concatenate(
                (self.datasets[time][self.mode]['text'], self.datasets[prev_time][self.mode]['text'][:-last_K_num_samples]), axis=0)
            self.datasets[time][self.mode]['labels'] = np.concatenate(
                (self.datasets[time][self.mode]['labels'], self.datasets[prev_time][self.mode]['labels'][:-last_K_num_samples]), axis=0)
            del self.datasets[prev_time]
            for classid in range(self.num_classes):
                sel_idx = np.nonzero(self.datasets[time][self.mode]['labels'] == classid)[0]
                self.class_id_list[classid][time] = sel_idx
        else:
            self.update_historical(idx)

    def update_current_timestamp(self, time):
        self.current_time = time

    def get_lisa_new_sample(self, time_idx, classid, num_sample):
        idx_all = self.class_id_list[classid][time_idx]
        if len(idx_all) == 0:
            return None, None
        sel_idx = np.random.choice(idx_all, num_sample, replace=True)[0]
        headline = self.datasets[time_idx][self.mode]['text'][sel_idx]
        category = self.datasets[time_idx][self.mode]['labels'][sel_idx]

        x = self.transform(text=headline)
        y = torch.LongTensor([category])

        return x.unsqueeze(0).cuda(), y.cuda()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'ecthr_a'


class ECtHRA(ECtHRABase):
    def __init__(self, args):
        super().__init__(args=args)

    def __getitem__(self, index):
        if self.args.difficulty and self.mode == 0:
            # Pick a time step from all previous timesteps
            idx = self.ENV.index(self.current_time)
            window = np.arange(0, idx + 1)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx, end_idx = self.task_idxs[sel_time][self.mode]

            # Pick an example in the time step
            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            index = sel_idx

        headline = self.datasets[self.current_time][self.mode]['text'][index]
        category = self.datasets[self.current_time][self.mode]['labels'][index]

        x = self.transform(text=headline)
        y = torch.LongTensor([category])

        return x, y

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])


class ECtHRAGroup(ECtHRABase):
    def __init__(self, args):
        super().__init__(args=args)
        self.num_groups = args.num_groups
        self.group_size = args.group_size
        self.window_end = self.ENV[0]
        self.train = True
        self.groupnum = 0

    def __getitem__(self, index):
        if self.mode == 0:
            np.random.seed(index)
            # Select group ID
            idx = self.ENV.index(self.current_time)
            if self.args.non_overlapping:
                possible_groupids = [i for i in range(0, max(1, idx - self.group_size + 1), self.group_size)]
                if len(possible_groupids) == 0:
                    possible_groupids = [np.random.randint(self.group_size)]
            else:
                possible_groupids = [i for i in range(max(1, idx - self.group_size + 1))]
            groupid = np.random.choice(possible_groupids)

            # Pick a time step in the sliding window
            window = np.arange(max(0, idx - groupid - self.group_size), idx + 1)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx = self.task_idxs[sel_time][0]
            end_idx = self.task_idxs[sel_time][1]

            # Pick an example in the time step
            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            headline = self.datasets[self.current_time][self.mode]['text'][sel_idx]
            category = self.datasets[self.current_time][self.mode]['labels'][sel_idx]
            x = self.transform(text=headline)
            y = torch.LongTensor(category)
            group_tensor = torch.LongTensor([groupid])

            del groupid
            del window
            del sel_time
            del start_idx
            del end_idx
            del sel_idx

            return x, y, group_tensor

        else:
            headline = self.datasets[self.current_time][self.mode]['text'][index]
            category = self.datasets[self.current_time][self.mode]['labels'][index]

            x = self.transform(text=headline)
            y = torch.LongTensor(category)

            return x, y

    def group_counts(self):
        idx = self.ENV.index(self.current_time)
        return torch.LongTensor([1 for _ in range(min(self.num_groups, idx + 1))])

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])


"""
Categories to IDs:
    {'10': 0, '11': 1, '13': 2, '14': 3, '2': 4, '3': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10,
    'P1-1': 11, 'P1-3': 12, 'P4-2': 13}
    
IDs to Categories:
    {0: '10', 1: '11', 2: '13', 3: '14', 4: '2', 5: '3', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9',
    11: 'P1-1', 12: 'P1-3', 13: 'P4-2'}
"""


def preprocess_orig(args):
    base_dfs = []
    for path in RAW_DATA_FILE:
        raw_data_path = os.path.join(args.data_dir, path)
        if not os.path.isfile(raw_data_path):
            raise ValueError(f'{path} is not in the data directory {args.data_dir}!')
        base_dfs.append(pd.read_json(raw_data_path, lines=True))
    # Load data frame from json file, group by year
    base_df = pd.concat(base_dfs)
    base_df = base_df.sort_values(by=['judgment_date'])
    allowed = ['10', '11', '13', '14', '2', '3', '5', '6', '7', '8', '9', 'P1-1', 'P1-3', 'P4-2']
    # allowed = ['10', '11', '13', '14', '18', '2', '3', '4', '5', '6', '7', '8', '9', 'P1-1', 'P4-2', 'P7-1', 'P7-4']
    for i in range(len(base_df)):
        new_label = []
        for label in base_df.iloc[i, 8]:
            if label in allowed:
                new_label.append(label)
        base_df.iloc[i, 8] = new_label
    base_df['year'] = pd.DatetimeIndex(base_df['judgment_date']).year
    df_years = base_df.groupby(pd.Grouper(key='year'))
    all_dfs = [group for _, group in df_years]
    all_years = list(base_df['year'].unique())
    dfs = []
    years = []
    if GROUP < 2:
        dfs = [pd.concat(all_dfs[:2])] + all_dfs[2:]
        years = all_years[1:]
    else:
        dfs.append(pd.concat(all_dfs[:2]))
        years.append(all_years[1])
        all_dfs = all_dfs[2:]
        all_years = all_years[2:]
        for i in range(math.ceil(len(all_years)/GROUP)):
            try:
                dfs.append(pd.concat(all_dfs[GROUP*i:GROUP*i+GROUP]))
                years.append(all_years[GROUP*i + 1])
            except:
                dfs.append(pd.concat(all_dfs[GROUP*i:]))
                years.append(all_years[-1])

    # Identify class ids that appear in all years 2012 - 2018
    categories_to_classids = {category: classid for classid, category in
                              enumerate(sorted(list(set([i for sublist in base_df['violated_articles'] for i in sublist]))))}
    classids_to_categories = {v: k for k, v in categories_to_classids.items()}

    dataset = {}
    for i, year in enumerate(years):
        if i == 0:
            continue
        # Store news headlines and category labels
        dataset[year] = {}
        df_year = dfs[i - 1]
        for j in df_year.index:
            new_label = []
            for label in df_year.at[j, 'violated_articles']:
                if label in categories_to_classids.keys():
                    new_label.append(label)
            df_year.at[j, 'violated_articles'] = new_label

        headlines = df_year['facts'].tolist()
        categories = []
        for label in df_year['violated_articles']:
            new_label = []
            for l in label:
                new_label.append(categories_to_classids[l])
            categories.append(new_label)
        headlines_train = pd.Series(headlines)
        categories_train = pd.Series(categories)

        df_year = dfs[i]
        for j in df_year.index:
            new_label = []
            for label in df_year.at[j, 'violated_articles']:
                if label in categories_to_classids.keys():
                    new_label.append(label)
            df_year.at[j, 'violated_articles'] = new_label

        headlines = df_year['facts'].tolist()
        categories = []
        for label in df_year['violated_articles']:
            new_label = []
            for l in label:
                new_label.append(categories_to_classids[l])
            categories.append(new_label)
        headlines_val = pd.Series(headlines)
        categories_val = pd.Series(categories)

        seed_ = np.random.get_state()
        np.random.seed(0)
        np.random.set_state(seed_)

        dataset[year][0] = {}
        dataset[year][0]['text'] = headlines_train.to_numpy()
        dataset[year][0]['labels'] = categories_train.to_numpy()
        dataset[year][1] = {}
        dataset[year][1]['text'] = headlines_val.to_numpy()
        dataset[year][1]['labels'] = categories_val.to_numpy()
        dataset[year][2] = {}
        dataset[year][2]['text'] = headlines_val.to_numpy()
        dataset[year][2]['labels'] = categories_val.to_numpy()

    preprocessed_data_path = os.path.join(args.data_dir, PREPROCESSED_FILE)
    pickle.dump(dataset, open(preprocessed_data_path, 'wb'))


def preprocess(args):
    np.random.seed(0)
    if not os.path.isfile(os.path.join(args.data_dir, PREPROCESSED_FILE)):
        preprocess_orig(args)
    np.random.seed(args.random_seed)

