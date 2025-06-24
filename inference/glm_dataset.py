import torch
from esm import FastaBatchedDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class FastaBatchedContigDataset(FastaBatchedDataset):
    """
    A dataset that extends FastaBatchedDataset to handle contig-to-protein mappings.

    Args:
        sequence_labels (list of str): Labels for each protein sequence.
        sequence_strs (list of str): Protein sequences.
        contig2prot_tsv (str, optional): Path to a TSV file mapping contigs to protein accessions.
    """
    def __init__(self, sequence_labels, sequence_strs, contig2prot_tsv=None):
        super().__init__(sequence_labels, sequence_strs)
        self.from_contig_file(contig2prot_tsv)

    def from_contig_file(self, contig2prot_tsv):
        """
        Processes the contig-to-protein mapping TSV file and builds lookup dictionaries.

        Args:
            contig2prot_tsv (str): Path to a TSV file where each line contains a contig
                                   and a semicolon-separated list of protein accessions
                                   prefixed with orientation ('+' or '-').
        """
        self.contig2prot_tsv = contig2prot_tsv
        # Map protein ID to sequence index
        self.prot2id = {label.split(' ')[0]: i for i, label in enumerate(self.sequence_labels)}
        # Reverse mapping from sequence index to protein ID
        self.id2prot = {v: k for k, v in self.prot2id.items()}
        # Orientation for each protein (e.g., '+', '-')
        self.prot_oris = {}
        # Mapping from contigs to list of protein sequence indices
        self.contigs = self._contig_to_prots()

    def _contig_to_prots(self):
        """
        Reads the TSV file and builds a dictionary mapping contigs to list of protein indices.
        Also populates orientation data for each protein.

        Returns:
            dict: Mapping of contig -> list of protein indices.
        """
        if self.contig2prot_tsv is None:
            return {}

        with open(self.contig2prot_tsv, 'r') as f:
            lines = f.readlines()

        contigs = {}
        for line in lines:
            contig, prots = line.strip().split("\t")
            prots_in_contig = prots.split(";")
            # Extract orientation and update orientation dictionary
            self.prot_oris.update({prot[1:]: prot[0] for prot in prots_in_contig})
            # Map contig to list of protein indices
            contigs[contig] = [self.prot2id[prot[1:]] for prot in prots_in_contig]
        return contigs

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset, including label, sequence, protein ID(s), and orientation(s).

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (label(s), sequence(s), protein ID(s), orientation(s))
        """
        labels = self.sequence_labels[idx]
        if isinstance(labels, list):
            prots = [label.split(' ')[0] for label in labels]
            ids = [self.prot2id[prot] for prot in prots]
            oris = [self.prot_oris[prot] for prot in prots]
        else:
            prots = labels.split(' ')[0]
            ids = self.prot2id[prots]
            oris = self.prot_oris[prots]
        return labels, self.sequence_strs[idx], ids, oris

def get_collate_fn(tokenizer, **kwargs):
    """
    Creates a custom collate function to prepare batches for the model using a tokenizer.

    Args:
        tokenizer (Callable): A tokenizer that accepts a list of sequences and returns encoded tensors.
        **kwargs: Additional keyword arguments for the tokenizer.

    Returns:
        Callable: A collate function to be used with a DataLoader.
    """
    def collate_fn(raw_batch):
        # Unzip the batch into separate lists
        batch_labels, seq_str_list, prot_ids, prot_oris = zip(*raw_batch)

        # Enable default padding if not provided
        if 'padding' not in kwargs.keys():
            kwargs['padding'] = True

        # If using fixed-length padding, also enable truncation
        if kwargs['padding'] == 'max_length':
            kwargs['truncation'] = True

        # Tokenize sequences
        inputs = tokenizer(seq_str_list, return_tensors='pt', **kwargs)
        # Add metadata to batch dictionary
        inputs.update({'prot_ids': prot_ids, 'prot_oris': prot_oris, 'labels': batch_labels})
        return inputs

    return collate_fn

def contig_collate_fn(sequences):
    """
    Custom collate function to pad batched sequence data and prepare it for training.

    Args:
        sequences (list of dict): List of sample dictionaries returned by ContigDataset.

    Returns:
        dict: A dictionary with the same keys as the input samples. All tensor fields are
              padded to the maximum sequence length in the batch (except 'prot_ids').
    """
    keys = sequences[0].keys()
    result = {}
    for k in keys:
        values = [seq[k] for seq in sequences]
        # Pad all tensor fields except 'prot_ids'
        if k != 'prot_ids':
            result[k] = pad_sequence(values, batch_first=True, padding_value=0)
        else:
            result[k] = pad_sequence([torch.as_tensor(v) for v in values], batch_first=True, padding_value=-1)
    return result

class ContigDataset(Dataset):
    """
    Dataset for contig-based protein embeddings with PCA and normalization applied.

    Args:
        cds: Object containing contig → protein index mappings.
        esm_embeds (list of tuples): Each tuple is (protein_id, orientation, embedding).
        norm_factors (str or dict): Normalization parameters with 'mean' and 'std'.
        pca_parms (str or dict): PCA parameters with 'components', 'mean', and 'explained_variance'.
        max_seq_len (int): Maximum sequence length (unused here, but stored).
    """
    def __init__(self, cds, esm_embeds, norm_factors, pca_parms, max_seq_len=30):
        self.apply_norm = self.get_apply_norm_fn(norm_factors)
        self.apply_pca, self.label_dim, self.emb_dim = self.get_apply_pca_fn(pca_parms)

        # Normalize embeddings and sort them by protein ID and orientation
        norm_embs = sorted([(prot_id, ori, self.apply_norm(emb))
                            for prot_id, ori, emb in esm_embeds])

        # Concatenate embedding with orientation indicator (+0.5 or -0.5)
        self.embs = torch.stack([
            torch.cat([emb, torch.as_tensor([.5 if ori == '+' else -.5])])
            for _, ori, emb in norm_embs
        ], dim=0)

        # Apply PCA to normalized embeddings (excluding orientation), and concatenate back
        self.labels = torch.cat([
            self.apply_pca(self.embs[:, :-1]),  # PCA-reduced features
            self.embs[:, -1:]                   # Orientation component
        ], dim=1)

        self.cds = cds
        self.contigs = list(cds.contigs.values())
        self.max_seq_len = max_seq_len
        self.keys = ['inputs_embeds', 'labels', 'attention_mask']

    @staticmethod
    def slice_to_indices(slice_obj, sequence_length):
        """
        Converts a Python slice object to a list of integer indices.

        Args:
            slice_obj (slice): Python slice object.
            sequence_length (int): Length of the sequence for resolving slice boundaries.

        Returns:
            list of int: Expanded indices.
        """
        return list(range(*slice_obj.indices(sequence_length)))

    @staticmethod
    def get_apply_pca_fn(parms):
        """
        Builds a PCA transformation function using the provided parameters.

        Args:
            parms (dict or str): Dictionary of PCA components, or path to a .pt file.

        Returns:
            tuple: (apply_pca_fn, output_dim, input_dim)
        """
        if isinstance(parms, str):
            parms = torch.load(parms)

        label_dim, emb_dim = parms['components'].shape

        def apply_pca(X):
            X_0 = X - parms['mean']
            X_transformed = torch.matmul(X_0, parms['components'].T)
            X_white = X_transformed / torch.sqrt(parms['explained_variance'])
            return X_white

        return apply_pca, label_dim + 1, emb_dim + 1  # +1 for orientation channel

    @staticmethod
    def get_apply_norm_fn(parms):
        """
        Builds a normalization function using the provided parameters.

        Args:
            parms (dict or str): Dictionary with 'mean' and 'std', or path to a .pt file.

        Returns:
            Callable: A function that applies standard normalization.
        """
        if isinstance(parms, str):
            parms = torch.load(parms)

        def apply_norm(X):
            X_0 = X - parms['mean']
            X_norm = X_0 / parms['std']
            return X_norm

        return apply_norm

    def __getitem__(self, idx):
        """
        Fetches a sample or batch from the dataset.

        Args:
            idx (int, slice, or list): Index, slice, or list of indices.

        Returns:
            dict: Dictionary containing:
                - 'inputs_embeds': (seq_len, emb_dim+1) tensor of inputs
                - 'labels': (seq_len, label_dim+1) tensor of PCA labels
                - 'attention_mask': 1s for valid positions
                - 'prot_ids': original protein indices
        """
        if isinstance(idx, slice):
            idx = self.slice_to_indices(idx, len(self.contigs))
        if isinstance(idx, list):
            sequences = [self[i] for i in idx]
            return contig_collate_fn(sequences)

        return {
            'inputs_embeds': self.embs[self.contigs[idx]],
            'labels': self.labels[self.contigs[idx]],
            'attention_mask': torch.ones((len(self.contigs[idx]),), dtype=torch.long),
            'prot_ids': self.contigs[idx]
        }

    def __len__(self):
        """
        Returns:
            int: Number of contigs in the dataset.
        """
        return len(self.contigs)
