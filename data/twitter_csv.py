"""
Twitter/Social Media CSV Dataset Loader for PRODIGY

This loader handles Twitter data with the following structure:
- Nodes: Users (userid)
- Edges: Social interactions (mentions, retweets, replies, quotes)
- Features: BERT-encoded text (descriptions, tweets)
- Labels: Configurable (verified, location, sentiment, etc.)

CSV Headers Expected:
tweetid, userid, screen_name, date, lang, location, description, place_id, place_url,
place_type, place_name, place_full_name, place_country_code, place_country,
place_bounding_box, text, extended, coord, reply_userid, reply_screen, reply_statusid,
tweet_type, friends_count, listed_count, followers_count, favourites_count,
statuses_count, verified, hashtag, urls_list, profile_pic_url, profile_banner_url,
display_name, date_first_tweet, account_creation_date, rt_urls_list, mentionid,
mentionsn, rt_screen, rt_userid, rt_user_description, rt_text, rt_hashtag,
rt_qtd_count, rt_rt_count, rt_reply_count, rt_fav_count, rt_tweetid, rt_location,
qtd_screen, qtd_userid, qtd_user_description, qtd_text, qtd_hashtag, qtd_qtd_count,
qtd_rt_count, qtd_reply_count, qtd_fav_count, qtd_tweetid, qtd_urls_list,
qtd_location, sent_vader, token, media_urls, rt_media_urls, q_media_urls, state,
country, rt_state, rt_country, qtd_state, qtd_country, norm_country, norm_rt_country,
norm_qtd_country, acc_age
"""

import os
from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Set, Any, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from experiments.sampler import NeighborSampler
from .augment import get_aug
from .dataloader import MulticlassTask, ParamSampler, BatchSampler, Collator
from .dataset import SubgraphDataset


def load_twitter_csv(
        csv_path: str,
        label_type: str = "verified",
        use_text_features: bool = False,
        bert: Optional[Any] = None,
        max_users: Optional[int] = None
) -> Data:
    """
    Load Twitter CSV and construct a user interaction graph.

    Args:
        csv_path: Path to CSV file
        label_type: What to use as labels. Options:
            - "verified": User verification status (binary)
            - "country": User country/location
            - "sentiment": Tweet sentiment (sent_vader)
            - "tweet_type": Type of tweet
            - "lang": Tweet language
        use_text_features: If True, use BERT to encode text features
        bert: BERT model for encoding (required if use_text_features=True)
        max_users: Maximum number of users to include (for testing on subset)

    Returns:
        PyTorch Geometric Data object
    """
    print(f"Loading Twitter CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} tweets")

    # Create user graph
    # Step 1: Build user index
    all_user_ids = set()
    all_user_ids.update(df['userid'].dropna().unique())

    # Add users from interactions
    all_user_ids.update(df['reply_userid'].dropna().unique())
    all_user_ids.update(df['rt_userid'].dropna().unique())
    all_user_ids.update(df['qtd_userid'].dropna().unique())

    # Add mentioned users
    for mentions in df['mentionid'].dropna():
        if isinstance(mentions, str):
            # Handle format like "[123, 456]" or "123,456"
            mentions = mentions.strip('[]').split(',')
            all_user_ids.update([int(m.strip()) for m in mentions if m.strip().isdigit()])

    all_user_ids = sorted(list(all_user_ids))

    if max_users is not None:
        all_user_ids = all_user_ids[:max_users]
        print(f"Limited to {max_users} users")

    print(f"Found {len(all_user_ids)} unique users")

    user_to_idx = {uid: idx for idx, uid in enumerate(all_user_ids)}
    idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}

    # Step 2: Build edges (user interactions)
    edges = []
    edge_types = []  # 0: mention, 1: retweet, 2: reply, 3: quote

    for _, row in df.iterrows():
        source_user = row['userid']
        if pd.isna(source_user) or source_user not in user_to_idx:
            continue
        source_idx = user_to_idx[source_user]

        # Mention edges
        if pd.notna(row['mentionid']):
            mentions = str(row['mentionid']).strip('[]').split(',')
            for mention in mentions:
                mention = mention.strip()
                if mention.isdigit():
                    target_user = int(mention)
                    if target_user in user_to_idx:
                        target_idx = user_to_idx[target_user]
                        edges.append([source_idx, target_idx])
                        edge_types.append(0)

        # Retweet edges
        if pd.notna(row['rt_userid']):
            target_user = row['rt_userid']
            if target_user in user_to_idx:
                target_idx = user_to_idx[target_user]
                edges.append([source_idx, target_idx])
                edge_types.append(1)

        # Reply edges
        if pd.notna(row['reply_userid']):
            target_user = row['reply_userid']
            if target_user in user_to_idx:
                target_idx = user_to_idx[target_user]
                edges.append([source_idx, target_idx])
                edge_types.append(2)

        # Quote tweet edges
        if pd.notna(row['qtd_userid']):
            target_user = row['qtd_userid']
            if target_user in user_to_idx:
                target_idx = user_to_idx[target_user]
                edges.append([source_idx, target_idx])
                edge_types.append(3)

    if len(edges) == 0:
        print("WARNING: No edges found! Creating a few random edges for testing...")
        # Create some random edges so graph isn't empty
        for i in range(min(100, len(all_user_ids) - 1)):
            edges.append([i, i + 1])
            edge_types.append(0)

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_types, dtype=torch.long)

    print(f"Created {len(edges)} edges")
    print(f"Edge types - Mentions: {(edge_attr == 0).sum()}, Retweets: {(edge_attr == 1).sum()}, "
          f"Replies: {(edge_attr == 2).sum()}, Quotes: {(edge_attr == 3).sum()}")

    # Step 3: Build node features
    # Aggregate user information from all their tweets
    user_data = defaultdict(lambda: {
        'descriptions': [],
        'texts': [],
        'verified': None,
        'country': None,
        'sentiment': [],
        'followers_count': 0,
        'friends_count': 0,
        'statuses_count': 0,
        'lang': None,
    })

    for _, row in df.iterrows():
        uid = row['userid']
        if pd.isna(uid) or uid not in user_to_idx:
            continue

        user = user_data[uid]

        # Collect text
        if pd.notna(row['description']):
            user['descriptions'].append(str(row['description']))
        if pd.notna(row['text']):
            user['texts'].append(str(row['text']))

        # Get user attributes (take first non-null value)
        if user['verified'] is None and pd.notna(row['verified']):
            user['verified'] = 1 if row['verified'] in [True, 'True', 'true', 1, '1'] else 0

        if user['country'] is None and pd.notna(row['norm_country']):
            user['country'] = str(row['norm_country'])
        elif user['country'] is None and pd.notna(row['country']):
            user['country'] = str(row['country'])

        if pd.notna(row['sent_vader']):
            try:
                user['sentiment'].append(float(row['sent_vader']))
            except:
                pass

        if user['lang'] is None and pd.notna(row['lang']):
            user['lang'] = str(row['lang'])

        # Update counts (use max)
        for count_field in ['followers_count', 'friends_count', 'statuses_count']:
            if pd.notna(row[count_field]):
                try:
                    user[count_field] = max(user[count_field], int(row[count_field]))
                except:
                    pass

    # Step 4: Create node features and labels
    if use_text_features and bert is not None:
        print("Encoding text features with BERT...")
        node_texts = []
        for uid in all_user_ids:
            user = user_data[uid]
            # Combine description and tweets
            desc = ' '.join(user['descriptions'][:3]) if user['descriptions'] else "User"
            tweets = ' '.join(user['texts'][:5]) if user['texts'] else ""
            combined = f"{desc}. {tweets}"[:512]  # Limit length
            node_texts.append(combined)

        # Encode with BERT
        x = bert.get_sentence_embeddings(node_texts)
    else:
        print("Using numerical features...")
        # Use numerical features
        features = []
        for uid in all_user_ids:
            user = user_data[uid]
            feat = [
                user['verified'] if user['verified'] is not None else 0,
                user['followers_count'],
                user['friends_count'],
                user['statuses_count'],
                np.mean(user['sentiment']) if user['sentiment'] else 0,
                len(user['texts']),  # Number of tweets
            ]
            features.append(feat)

        x = torch.tensor(features, dtype=torch.float)

        # Normalize
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)

        # Pad to 768 dimensions (BERT-like) by repeating
        target_dim = 768
        if x.shape[1] < target_dim:
            repeats = target_dim // x.shape[1] + 1
            x = x.repeat(1, repeats)[:, :target_dim]

    # Step 5: Create labels based on label_type
    labels, label_names = create_labels(all_user_ids, user_data, label_type)

    y = torch.tensor(labels, dtype=torch.long)

    print(f"Label type: {label_type}")
    print(f"Number of classes: {len(label_names)}")
    print(f"Label distribution: {[(name, (y == i).sum().item()) for i, name in enumerate(label_names)]}")

    # Step 6: Create graph
    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=len(all_user_ids)
    )

    # Store metadata
    graph.label_names = label_names
    graph.label_type = label_type
    graph.user_to_idx = user_to_idx
    graph.idx_to_user = idx_to_user

    return graph


def create_labels(
        all_user_ids: List[int],
        user_data: Dict[int, Dict[str, Any]],
        label_type: str
) -> Tuple[List[int], List[str]]:
    """Create labels based on label_type."""

    if label_type == "verified":
        # Binary: verified or not
        labels = [user_data[uid]['verified'] if user_data[uid]['verified'] is not None else 0
                  for uid in all_user_ids]
        label_names = ["Not Verified", "Verified"]

    elif label_type == "country":
        # Multi-class: countries
        countries = set()
        for uid in all_user_ids:
            if user_data[uid]['country']:
                countries.add(user_data[uid]['country'])

        countries = sorted(list(countries))
        country_to_idx = {c: i for i, c in enumerate(countries)}
        country_to_idx['unknown'] = len(countries)

        labels = []
        for uid in all_user_ids:
            country = user_data[uid]['country']
            if country and country in country_to_idx:
                labels.append(country_to_idx[country])
            else:
                labels.append(country_to_idx['unknown'])

        label_names = countries + ['unknown']

    elif label_type == "sentiment":
        # 3-class: negative, neutral, positive (based on average sentiment)
        labels = []
        for uid in all_user_ids:
            sentiments = user_data[uid]['sentiment']
            if sentiments:
                avg_sent = np.mean(sentiments)
                if avg_sent < -0.05:
                    labels.append(0)  # Negative
                elif avg_sent > 0.05:
                    labels.append(2)  # Positive
                else:
                    labels.append(1)  # Neutral
            else:
                labels.append(1)  # Default to neutral

        label_names = ["Negative", "Neutral", "Positive"]

    elif label_type == "lang":
        # Multi-class: languages
        languages = set()
        for uid in all_user_ids:
            if user_data[uid]['lang']:
                languages.add(user_data[uid]['lang'])

        languages = sorted(list(languages))
        lang_to_idx = {l: i for i, l in enumerate(languages)}
        lang_to_idx['unknown'] = len(languages)

        labels = []
        for uid in all_user_ids:
            lang = user_data[uid]['lang']
            if lang and lang in lang_to_idx:
                labels.append(lang_to_idx[lang])
            else:
                labels.append(lang_to_idx['unknown'])

        label_names = languages + ['unknown']

    elif label_type == "activity":
        # 3-class: low, medium, high activity (based on tweet count)
        tweet_counts = [len(user_data[uid]['texts']) for uid in all_user_ids]
        q33, q66 = np.percentile(tweet_counts, [33, 66])

        labels = []
        for uid in all_user_ids:
            count = len(user_data[uid]['texts'])
            if count <= q33:
                labels.append(0)
            elif count <= q66:
                labels.append(1)
            else:
                labels.append(2)

        label_names = ["Low Activity", "Medium Activity", "High Activity"]

    else:
        raise ValueError(f"Unknown label_type: {label_type}")

    return labels, label_names


def get_twitter_dataset(
        root: str,
        csv_filename: str = "twitter_data.csv",
        label_type: str = "verified",
        n_hop: int = 2,
        bert: Optional[Any] = None,
        bert_device: str = "cpu",
        original_features: bool = True,
        max_users: Optional[int] = None,
        **kwargs
) -> SubgraphDataset:
    """
    Load Twitter CSV dataset and create SubgraphDataset.

    Args:
        root: Root directory containing CSV
        csv_filename: Name of CSV file
        label_type: Type of labels to use (verified, country, sentiment, lang, activity)
        n_hop: Number of hops for neighbor sampling
        bert: BERT model for text encoding
        bert_device: Device for BERT
        original_features: If True, use numerical features; if False, use BERT text features
        max_users: Maximum users to include (for testing on subset)

    Returns:
        SubgraphDataset object
    """
    csv_path = os.path.join(root, csv_filename)

    # Cache path
    cache_name = f"twitter_graph_{label_type}"
    if not original_features:
        cache_name += "_bert"
    if max_users:
        cache_name += f"_maxusers{max_users}"
    cache_name += ".pt"
    cache_path = os.path.join(root, cache_name)

    # Load or create graph
    if os.path.exists(cache_path):
        print(f"Loading cached graph from {cache_path}")
        graph = torch.load(cache_path)
    else:
        print(f"Creating graph from CSV...")
        os.makedirs(root, exist_ok=True)

        use_text = not original_features and bert is not None
        graph = load_twitter_csv(csv_path, label_type=label_type,
                                 use_text_features=use_text, bert=bert,
                                 max_users=max_users)

        print(f"Caching graph to {cache_path}")
        torch.save(graph, cache_path)

    print(f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")

    # Create neighbor sampler
    neighbor_sampler = NeighborSampler(graph, num_hops=n_hop)

    return SubgraphDataset(graph, neighbor_sampler, bidirectional=False)


def twitter_task(
        root: str,
        csv_filename: str,
        label_type: str,
        split: str,
        label_set: Optional[Set[int]] = None,
        split_labels: bool = True,
        train_cap: Optional[int] = None,
        linear_probe: bool = False,
        max_users: Optional[int] = None
) -> MulticlassTask:
    """Define few-shot learning task for Twitter data."""

    # Load graph to get labels
    cache_name = f"twitter_graph_{label_type}"
    if max_users:
        cache_name += f"_maxusers{max_users}"
    cache_name += ".pt"
    cache_path = os.path.join(root, cache_name)

    if not os.path.exists(cache_path):
        # Need to create it first
        csv_path = os.path.join(root, csv_filename)
        graph = load_twitter_csv(csv_path, label_type=label_type,
                                 use_text_features=False, bert=None,
                                 max_users=max_users)
        torch.save(graph, cache_path)
    else:
        graph = torch.load(cache_path)

    labels = graph.y.numpy()
    num_classes = len(graph.label_names)

    print(f"Task - {num_classes} classes: {graph.label_names}")

    if label_set is not None:
        label_set = set(label_set)
    elif split_labels:
        # Meta-learning: split labels
        all_labels = list(range(num_classes))

        # 60% train, 20% val, 20% test
        n_train = max(1, int(num_classes * 0.6))
        n_val = max(1, int(num_classes * 0.2))

        TRAIN_LABELS = all_labels[:n_train]
        VAL_LABELS = all_labels[n_train:n_train + n_val]
        TEST_LABELS = all_labels[n_train + n_val:]

        # Ensure at least one label per split
        if not TEST_LABELS and VAL_LABELS:
            TEST_LABELS = [VAL_LABELS.pop()]
        if not VAL_LABELS and TRAIN_LABELS:
            VAL_LABELS = [TRAIN_LABELS.pop()]

        print(f"Split - Train: {len(TRAIN_LABELS)}, Val: {len(VAL_LABELS)}, Test: {len(TEST_LABELS)}")

        if split == "train":
            label_set = set(TRAIN_LABELS)
        elif split == "val":
            label_set = set(VAL_LABELS)
        elif split == "test":
            label_set = set(TEST_LABELS)
        else:
            raise ValueError(f"Invalid split: {split}")
    else:
        label_set = set(range(num_classes))

    # Handle train_cap
    train_label = None
    if train_cap is not None and split == "train":
        train_label = labels.copy()
        for i in range(num_classes):
            idx = np.where(labels == i)[0]
            if len(idx) > train_cap:
                disabled_idx = idx[train_cap:]
                train_label[disabled_idx] = -1 - i

    return MulticlassTask(labels, label_set, train_label, linear_probe)


def get_twitter_dataloader(
        dataset: SubgraphDataset,
        split: str,
        node_split: str,
        batch_size: Union[int, range],
        n_way: Union[int, range],
        n_shot: Union[int, range],
        n_query: Union[int, range],
        batch_count: int,
        root: str,
        bert: Optional[Any],
        num_workers: int,
        aug: str,
        aug_test: bool,
        split_labels: bool,
        train_cap: Optional[int],
        linear_probe: bool,
        label_set: Optional[Set[int]] = None,
        csv_filename: str = "twitter_data.csv",
        label_type: str = "verified",
        max_users: Optional[int] = None,
        **kwargs
) -> DataLoader:
    """Create DataLoader for Twitter few-shot tasks."""

    # Get label embeddings from label names
    graph = dataset.graph
    label_names = graph.label_names

    if bert is not None:
        label_embeddings = bert.get_sentence_embeddings(label_names)
    else:
        # Fallback: random embeddings
        label_embeddings = torch.randn(len(label_names), 768)

    # Create task
    task = twitter_task(root, csv_filename, label_type, split, label_set,
                        split_labels, train_cap, linear_probe, max_users)

    # Create sampler
    sampler = BatchSampler(
        batch_count, task,
        ParamSampler(batch_size, n_way, n_shot, n_query, 1),
        seed=42,
    )

    # Augmentation
    if split == "train" or aug_test:
        aug_fn = get_aug(aug, dataset.graph.x)
    else:
        aug_fn = get_aug("")

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=num_workers,
        collate_fn=Collator(label_embeddings, aug=aug_fn)
    )

    return dataloader


# Example usage
if __name__ == "__main__":
    """Test the Twitter dataset loader."""
    print("Testing Twitter CSV loader...")

    # You'll need to update this path to your actual CSV
    root = "./data/twitter"
    csv_file = "twitter_data.csv"

    # Test without BERT first (faster)
    print("\n=== Test 1: Loading with numerical features ===")
    dataset = get_twitter_dataset(
        root=root,
        csv_filename=csv_file,
        label_type="verified",  # Try: verified, country, sentiment, lang, activity
        n_hop=2,
        original_features=True,
        max_users=1000  # Limit for testing
    )

    print(f"Dataset created: {len(dataset)} nodes")
    print(f"Graph info: {dataset.graph}")

    # Test with BERT (if available)
    # print("\n=== Test 2: Loading with BERT text features ===")
    # from models.sentence_embedding import SentenceEmb
    # bert = SentenceEmb("sentence-transformers/all-mpnet-base-v2", device="cpu")
    #
    # dataset_bert = get_twitter_dataset(
    #     root=root,
    #     csv_filename=csv_file,
    #     label_type="sentiment",
    #     n_hop=2,
    #     bert=bert,
    #     original_features=False,
    #     max_users=1000
    # )

    print("\nTest complete!")
