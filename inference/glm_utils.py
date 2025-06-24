import torch
import torch.nn as nn
import os
import numpy as np

def save_results(batch, outputs, folder, prefix):
    if not os.path.exists(folder):
        os.mkdir(folder)
        
    # converts to numpy and fp16
    input_embs = batch['inputs_embeds'].numpy().astype(np.float16)
    label_embs = batch['labels'].numpy().astype(np.float16)
    hidden_embs = outputs.last_hidden_state.numpy().astype(np.float16)
    output_embs = outputs.logits_all_preds.numpy().astype(np.float16)
    probs = nn.Softmax(dim=1)(outputs.probs.view(-1,4))
    all_probs = probs.numpy().astype(np.float16)
    all_prot_ids = batch['prot_ids'].numpy().astype(np.int8)
    all_contacts = outputs.contacts.numpy().astype(np.float16)

    # concatenates batches together
    input_embs = np.concatenate(input_embs, axis = 0)
    label_embs = np.concatenate(label_embs, axis = 0)
    hidden_embs = np.concatenate(hidden_embs, axis = 0)
    output_embs = np.concatenate(output_embs, axis = 0)
    all_probs = np.concatenate(all_probs, axis = 0)
    all_prot_ids = np.concatenate(all_prot_ids, axis = 0)
    all_contacts = np.concatenate(all_contacts, axis =0)

    # removes padded proteins
    input_embs = input_embs[np.where(all_prot_ids != -1)[0]]
    label_embs = label_embs[np.where(all_prot_ids != -1)[0]]
    hidden_embs = hidden_embs[np.where(all_prot_ids != -1)[0]]
    output_embs = output_embs[np.where(all_prot_ids != -1)[0]]
    all_probs = all_probs[np.where(all_prot_ids != -1)[0]]
    all_prot_ids = all_prot_ids[np.where(all_prot_ids != -1)[0]]

    results = {}
    results['plm_embs'] = input_embs
    results['label_embs'] = label_embs
    results['glm_embs'] = hidden_embs
    results['output_embs'] = output_embs
    results['all_probs'] = all_probs
    results['all_prot_ids'] = all_prot_ids

    glm_embs = []
    for i, emb in enumerate(hidden_embs):
        glm_embs.append((all_prot_ids[i], emb))

    torch.save(results, os.path.join(folder, f'{prefix}_results.pt'))
    torch.save(glm_embs, os.path.join(folder, f'{prefix}_glm_embs.pt'))
    torch.save(all_contacts, os.path.join(folder, f'{prefix}_attention.pt'))