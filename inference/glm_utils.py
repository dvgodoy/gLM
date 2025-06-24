import torch
import torch.nn as nn
import os
import numpy as np

def save_results(batches, outputs, folder, prefix):
    """
    Saves model embeddings, predictions, and associated metadata from a batches to disk.

    Args:
        batches (Dict[str, torch.Tensor]): Dictionary containing input batches tensors such as
                                         'inputs_embeds', 'labels', and 'prot_ids'.
        outputs (Dict[str, torch.Tensor]): Dictionary containing model output tensors such as
                                           'last_hidden_state', 'logits_all_preds', 'probs', 'contacts'.
        folder (str): Path to the directory where result files will be saved.
        prefix (str): Prefix for the saved file names.

    Saves:
        - {prefix}_results.pt: Dictionary with input, label, hidden, output embeddings, probabilities, and IDs.
        - {prefix}_glm_embs.pt: List of tuples (protein_id, embedding) for each protein.
        - {prefix}_attention.pt: Raw contact map or attention data.
    """
    if not os.path.exists(folder):
        os.mkdir(folder)

    # Convert tensors to NumPy arrays and cast to float16 to save space
    input_embs = batches['inputs_embeds'].numpy().astype(np.float16)
    label_embs = batches['labels'].numpy().astype(np.float16)
    hidden_embs = outputs['last_hidden_state'].numpy().astype(np.float16)
    output_embs = outputs['logits_all_preds'].numpy().astype(np.float16)

    # Compute softmax probabilities and convert to NumPy
    probs = nn.Softmax(dim=1)(outputs['probs'].view(-1, 4))
    all_probs = probs.numpy().astype(np.float16)

    # Convert protein IDs and contacts
    all_prot_ids = batches['prot_ids'].numpy().astype(int)
    all_contacts = outputs['contacts'].numpy().astype(np.float16)

    # Concatenate tensors along the batches dimension
    input_embs = np.concatenate(input_embs, axis=0)
    label_embs = np.concatenate(label_embs, axis=0)
    hidden_embs = np.concatenate(hidden_embs, axis=0)
    output_embs = np.concatenate(output_embs, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_prot_ids = np.concatenate(all_prot_ids, axis=0)
    all_contacts = np.concatenate(all_contacts, axis=0)

    # Remove entries corresponding to padded proteins (ID == -1)
    valid_idx = np.where(all_prot_ids != -1)[0]
    input_embs = input_embs[valid_idx]
    label_embs = label_embs[valid_idx]
    hidden_embs = hidden_embs[valid_idx]
    output_embs = output_embs[valid_idx]
    all_probs = all_probs[valid_idx]
    all_prot_ids = all_prot_ids[valid_idx].tolist()

    # Package the results into a dictionary
    results = {
        'plm_embs': input_embs,
        'label_embs': label_embs,
        'glm_embs': hidden_embs,
        'output_embs': output_embs,
        'all_probs': all_probs,
        'all_prot_ids': all_prot_ids,
    }

    # Create a list of tuples: (protein_id, hidden_state_embedding)
    glm_embs = [(all_prot_ids[i], emb) for i, emb in enumerate(hidden_embs)]

    # Save the structured results and attention/contact maps
    torch.save(results, os.path.join(folder, f'{prefix}_results.pt'))
    torch.save(glm_embs, os.path.join(folder, f'{prefix}_glm_embs.pt'))
    torch.save(all_contacts, os.path.join(folder, f'{prefix}_attention.pt'))
