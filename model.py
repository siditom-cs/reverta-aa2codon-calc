import os
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration, LogitsWarper
from transformers import LogitsProcessor,LogitsProcessorList
import wget
import json

models = {}
def load_model(model_type='mimic', winsize=10):
    global models
    key = model_type+str(winsize)
    if key in models.keys():
        return models[key]

    tokenizer = AutoTokenizer.from_pretrained('siditom/tokenizer-codon_optimization-refined_expr')
    if os.path.exists("mask_restrict_dict.json"):
        os.remove("mask_restrict_dict.json")
    wget.download("https://huggingface.co/siditom/tokenizer-codon_optimization-refined_expr/resolve/main/mask_restrict_dict.json")
    mask_restrict_dict = {}
    with open('mask_restrict_dict.json','r') as handle:
        mask_restrict_dict = json.load(handle)
    model = BartForConditionalGeneration.from_pretrained("siditom/co-model_"+model_type+"-rexpr-"+str(winsize)+"w_2ft")
    model_info = {"model": model, "tokenizer": tokenizer, "restrict_dict": mask_restrict_dict}
    models[key] = model_info
    return model_info


class RestrictToAaLogitsWarper(LogitsWarper):
    def __init__(self, masked_input_ids: torch.LongTensor, restrict_dict: dict, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        self.masked_input_ids = masked_input_ids
        self.restrict_dict = restrict_dict
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]-1
        vocab_size = scores.shape[-1]
        if self.masked_input_ids.shape[-1] <= cur_len:
            return scores
        for bid in range(input_ids.shape[0]):
            cur_mask_input = str(int(self.masked_input_ids[bid][cur_len].item()))
            if cur_mask_input in self.restrict_dict.keys():
                restricted_words = self.restrict_dict[cur_mask_input]
                banned_indices = set(range(vocab_size))-set(self.restrict_dict[cur_mask_input])
                banned = torch.tensor([i in banned_indices for i in range(vocab_size)])
                scores[bid][banned] = self.filter_value
        return scores


def get_expr_token(expr):
    if expr==None:
        expr = 'expr_unk'
    return "<"+expr+"> "


def prepare_inputs(example, tokenizer, config):
        #Rename
        qaaseq, query_species, sw_aa_size = example['qseq'], example['query_species'],  config['sw_aa_size']
        if config['mimic']:
          sseq, subject_species = example['subject_dna_seq'].split(" "), example['subject_species']
        else:
          sseq, subject_species = ['<mask_'+aa+'>' if aa!='-' else '<gap>' for aa in qaaseq], example['query_species']


        #Prepare fixed-sized windows
        query_aa_wins = [qaaseq[i:i+sw_aa_size] for i in range(0,max(1,len(qaaseq)-sw_aa_size+1))]
        subject_dna_wins = [" ".join(sseq[i:i+sw_aa_size]) for i in range(0,max(1,len(qaaseq)-sw_aa_size+1))]
        mask_aa_wins = ["<"+query_species+"> "+" ".join(['<mask_'+aa+'>' if aa!='-' else '<gap>' for aa in wseq]) for wseq in query_aa_wins]
        query_aa_wins = ["<"+query_species+"> "+get_expr_token(example['expr'])+' '.join(['<mask_'+aa+'>' if aa!='-' else '<gap>' for aa in wseq]) for wseq in query_aa_wins]
        subject_dna_wins = ["<"+subject_species+"> "+wseq for wseq in subject_dna_wins]

        #Encode windows
        input_ids = tokenizer(subject_dna_wins, query_aa_wins, return_tensors="pt", padding='max_length', max_length=sw_aa_size*2+3).input_ids
        masked_ids = tokenizer(mask_aa_wins, return_tensors="pt").input_ids[:,1:-1]
        return input_ids,masked_ids


def generate_outputs(input_ids, masked_ids, mask_restriction_dict, model, sw_aa_size):
        logits_processor = LogitsProcessorList(
                [RestrictToAaLogitsWarper(masked_ids, mask_restriction_dict)])

        outputs = model.generate(input_ids, do_sample=False, output_scores = True, return_dict_in_generate = True, renormalize_logits = True, logits_processor=logits_processor, max_length=min((sw_aa_size+3),masked_ids.shape[-1]+2))
        outputs = torch.stack(outputs['scores'][:sw_aa_size+1],1)
        return outputs


def calc_combined_gen_from_sliding_windows_logits(sw_logits, seqlen, sw_aa_size):
        sw_logits = sw_logits
        collect_logits = torch.zeros([seqlen, sw_logits.shape[-1]])
        counts = torch.zeros([1,seqlen])
        most_freq_pred = torch.zeros([seqlen,1])

        #Aggregating (sums) the logits of the different windows. Only the relevant codons (restricted by AA) are sumed.
        for i in range(sw_logits.shape[0]): # window num
            for j in range(min(sw_aa_size, seqlen)): # sequence len - codon index
                collect_logits[i+j, :] += torch.exp(sw_logits[i, 1+j, :])
                counts[0,i+j] += 1

        #Normalizing each position by the number of predictions (eg. first codon has only one prediction)
        for i in range(seqlen):
            collect_logits[i,:] /= counts[0,i]
        collect_logits = torch.log(collect_logits)
        collect_logits = collect_logits.log_softmax(dim=-1)

        for i in range(seqlen):
            most_freq_pred[i] = torch.argmax(collect_logits[i,:]).item()

        return collect_logits, most_freq_pred


def predict(config, example, mask_restriction_dict, tokenizer, model):
        input_ids, masked_ids = prepare_inputs(example, tokenizer, config)
        outputs = generate_outputs(input_ids, masked_ids, mask_restriction_dict, model, config['sw_aa_size'])
        logits, most_freq_pred = calc_combined_gen_from_sliding_windows_logits(outputs, len(example['qseq']), config['sw_aa_size'])

        ce = torch.nn.CrossEntropyLoss()
        most_freq_pred=most_freq_pred.clone().detach().reshape((1,-1))


        #print("decode: ", tokenizer.decode(most_freq_pred.numpy().astype(int)[0]))
        #print("truevals: ", tokenizer.decode(true_vals))
        res = dict()

        res['prot_len'] = len(example['qseq'])
        res['prot_AAs'] = example['qseq']
        res['pred_codons'] = tokenizer.decode(most_freq_pred.numpy().astype(int)[0])
        res['entropy'] = (-torch.nan_to_num(torch.exp(logits)*logits,nan=0.0).sum(dim=-1)).mean().item()

        assert(res['prot_len']==len(res['pred_codons'].split(" ")))

        if config['calc_stats'] and 'query_dna_seq' in example.keys():
          true_vals = tokenizer(example['query_dna_seq'], return_tensors="pt").input_ids[:,1:-1]
          mask = true_vals > config['special_token_th']
          true_vals = true_vals.tolist()[0]
          masked_most_freq_pred = most_freq_pred.masked_select(mask).numpy().astype(int)
          masked_true_vals = torch.tensor(true_vals).masked_select(mask).numpy().astype(int)

          res['subject_codons'] = example['subject_dna_seq']
          res['num_of_correct_predicted_codons'] = sum([int(x==y) for x,y in zip(masked_true_vals, masked_most_freq_pred)])
          res['query_codons'] = example['query_dna_seq']
          res['cross_entropy_loss'] = ce(logits, torch.tensor(true_vals)).item()
          res['accuracy'] = res['num_of_correct_predicted_codons'] / res['prot_len']
        #print(example['qseqid'], example['sseqid'],res['cross_entropy_loss'], res['entropy'],res['accuracy'])
        return res

