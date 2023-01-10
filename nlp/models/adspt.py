import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers import AutoTokenizer, RobertaModel, RobertaPreTrainedModel, BertPreTrainedModel, RobertaForMaskedLM, AutoModel, AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from transformers.activations import gelu
from easydl import *

from models.udanli import GRL
from models.bert import CLS
class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                mask_token_id: int,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        """appends learned embedding to 
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.mask_token_id = mask_token_id
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                               n_tokens, 
                                                                               random_range, 
                                                                               initialize_from_vocab))
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
            
    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        mask_locs = (tokens == self.mask_token_id).nonzero(as_tuple=True)[1]
        input_embedding = self.wte(tokens)

        for i, _ in enumerate(tokens):
            # replace placeholder tokens with soft prompt
            input_embedding[i, mask_locs[i]-self.n_tokens:mask_locs[i]] = self.learned_embedding
        return input_embedding
class CLS(nn.Module):
    """
    a two-layer MLP for classification
    """
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim)
        self.main = nn.Sequential(self.bottleneck, self.fc) # output logits

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out
class AdversarialNetwork2(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork2, self).__init__()
        self.mid_dim = in_feature // 2
        self.main = nn.Sequential(
            nn.Linear(in_feature, self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.mid_dim, 1),
            # nn.Sigmoid() # output logits
        )
        self.grl = GRL()

    def forward(self, x):
        x_ = self.grl.apply(x)
        y = self.main(x_)
        return y

            
            
class RoBERTa_AdSPT_mlm_only_old(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, pretrained_model_name_or_path, config, num_domains, n_tokens, initialize_from_vocab=True):
        super().__init__(config)
        self.label_words = ['bad', 'good']
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.label_ids = [x[0] for x in self.tokenizer(self.label_words, add_special_tokens=False)['input_ids']]
        self.num_domains = num_domains
        pretrained_model = RobertaForMaskedLM.from_pretrained(pretrained_model_name_or_path)
        self.roberta = pretrained_model.roberta
        # RobertaModel.from_pretrained(pretrained_model_name_or_path, add_pooling_layer=False)
        self.lm_head = pretrained_model.lm_head    

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()
        s_wte = SoftEmbedding(self.roberta.get_input_embeddings(), 
                            mask_token_id=self.tokenizer.mask_token_id,
                            n_tokens=n_tokens, 
                            initialize_from_vocab=initialize_from_vocab)
        self.set_input_embeddings(s_wte)

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        domain_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        prediction_scores = outputs[0]
        mask_token_indices = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        mask_output = prediction_scores[list(range(prediction_scores.shape[0])), mask_token_indices, :]
        mask_scores = self.lm_head(mask_output)
        label_word_scores = torch.stack([mask_scores[i, id] for i in range(len(mask_token_indices)) for id in self.label_ids]).view(-1, len(self.label_ids))
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            masked_lm_loss = loss_fct(label_word_scores, labels.view(-1))
        else:
            masked_lm_loss = None
            

        if not return_dict:
            # output = (prediction_scores,) + outputs[2:]
            output = (label_word_scores, ) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            # logits=prediction_scores,
            logits=label_word_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    
class RoBERTa_AdSPT_mlm_only(RobertaPreTrainedModel):

    def __init__(self, pretrained_model_name_or_path, config, n_tokens, initialize_from_vocab=True, use_soft_prompt=True):
        super().__init__(config)
        self.label_words = [' bad', ' good']
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        # self.label_ids = [x[0] for x in self.tokenizer(self.label_words, add_special_tokens=False)['input_ids']]
        self.label_ids = [1099, 205]
        self.roberta = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path)

        # Initialize weights and apply final processing
        if use_soft_prompt:
            s_wte = SoftEmbedding(self.roberta.get_input_embeddings(), 
                                mask_token_id=self.tokenizer.mask_token_id,
                                n_tokens=n_tokens, 
                                initialize_from_vocab=initialize_from_vocab)
            self.set_input_embeddings(s_wte)

        for n, p in self.named_parameters():
            # if n.startswith('roberta.roberta.embeddings.') and not n.startswith('roberta.roberta.embeddings.word_embeddings.learned_embedding'):
            #     p.requires_grad = False
            print(f'{n} requires grad: {p.requires_grad}')
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs.logits
        mask_token_indices = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        mask_scores = logits[mask_token_indices[0], mask_token_indices[1], :]
        # label_word_scores = torch.stack([mask_scores[i, id] for i in range(len(mask_token_indices)) for id in self.label_ids]).view(-1, len(self.label_ids))
        label_word_scores = torch.stack([mask_scores[i, self.label_ids] for i in range(len(mask_token_indices[0]))])

        if labels is not None:
            # loss_fct = nn.BCEWithLogitsLoss()
            # masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            # masked_lm_loss = loss_fct(label_word_scores, F.one_hot(labels, num_classes=2).float())
            # nn.CrossEntropyLoss()(label_word_scores, labels.view(-1))
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(label_word_scores, labels.view(-1))
        else:
            masked_lm_loss = None
            

        if not return_dict:
            # output = (prediction_scores,) + outputs[2:]
            output = (label_word_scores, ) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            # logits=prediction_scores,
            logits=label_word_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    
class RoBERTa_AdSPT_single(RobertaPreTrainedModel):

    def __init__(self, pretrained_model_name_or_path, config, args, initialize_from_vocab=True, loss_coeff=None):
        super().__init__(config)
        self.label_words = [' bad', ' good']
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.label_ids = [1099, 205]
        robertaMLM = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path)
        self.roberta = robertaMLM.roberta ## DO NOT CHANGE THE NAME OF THE VARIABLE
        self.lm_head = robertaMLM.lm_head
        self.args = args
        self.discriminator = AdversarialNetwork2(self.roberta.config.hidden_size)
        print(f'{loss_coeff=}')
        if loss_coeff is None:
            loss_coeff = self.args.train.lr/self.args.train.plm_lr
        self.mul_grad = GradientReverseModule(lambda step: -loss_coeff) # lambda


        # Initialize weights and apply final processing
        s_wte = SoftEmbedding(self.roberta.get_input_embeddings(), 
                            mask_token_id=self.tokenizer.mask_token_id,
                            n_tokens=args.train.n_tokens, 
                            initialize_from_vocab=initialize_from_vocab)

        self.set_input_embeddings(s_wte)
        print(f'Initialized adspt with single source domain')
        # for n, p in self.named_parameters():
        #     # if n.startswith('roberta.roberta.embeddings.') and not n.startswith('roberta.roberta.embeddings.word_embeddings.learned_embedding'):
        #     #     p.requires_grad = False
        #     print(f'{n} requires grad: {p.requires_grad}')

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        mask_token_indices = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        mask_output = sequence_output[mask_token_indices[0], mask_token_indices[1], :]
        mask_scores = self.lm_head(self.mul_grad(mask_output))
        label_word_scores = torch.stack([mask_scores[i, self.label_ids] for i in range(len(mask_token_indices[0]))])
        # label_word_scores = torch.stack([mask_scores[i, id] for i in range(len(mask_token_indices)) for id in self.label_ids]).view(-1, len(self.label_ids))
        
        masked_lm_loss = None
        # if labels is not None:
            # loss_fct = nn.BCEWithLogitsLoss()
            # masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            # masked_lm_loss = loss_fct(label_word_scores, F.one_hot(labels, num_classes=2).float())
            # loss_fct = nn.CrossEntropyLoss()
            # masked_lm_loss = loss_fct(label_word_scores, labels.view(-1))
            
        # disc_input = self.grl(1, mask_output)
        # disc_input = self.grl(mask_output)
        discriminator_output = self.discriminator(mask_output) # (batch_size, 1)
        # discriminator_output = self.discriminator(disc_input) # (batch_size, 1)
        # for m_output, domain_id in zip(mask_output, domain_ids):
        #     discriminator_outputs[domain_id].append(self.discriminators[domain_id](m_output))
        # for i in range(len(discriminator_outputs)):
        #     discriminator_outputs[i] = torch.cat(discriminator_outputs[i], 0)

        if not return_dict:
            # output = (prediction_scores,) + outputs[2:]
            output = (label_word_scores, discriminator_output) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            # logits=prediction_scores,
            logits=label_word_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), discriminator_output



class BERT_AdSPT_single(BertPreTrainedModel):

    def __init__(self, pretrained_model_name_or_path, config, args, initialize_from_vocab=True, loss_coeff=None):
        super().__init__(config)
        self.label_words = [' bad', ' good']
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.label_ids = [1099, 205]
        MLM = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path)
        self.bert = MLM.bert ## DO NOT CHANGE THE NAME OF THE VARIABLE
        self.lm_head = MLM.lm_head
        self.args = args
        self.discriminator = AdversarialNetwork2(self.bert.config.hidden_size)
        print(f'{loss_coeff=}')
        if loss_coeff is None:
            loss_coeff = self.args.train.lr/self.args.train.plm_lr
        self.mul_grad = GradientReverseModule(lambda step: -loss_coeff) # lambda


        # Initialize weights and apply final processing
        s_wte = SoftEmbedding(self.bert.get_input_embeddings(), 
                            mask_token_id=self.tokenizer.mask_token_id,
                            n_tokens=args.train.n_tokens, 
                            initialize_from_vocab=initialize_from_vocab)

        self.set_input_embeddings(s_wte)
        print(f'Initialized adspt with single source domain')
        # for n, p in self.named_parameters():
        #     # if n.startswith('roberta.roberta.embeddings.') and not n.startswith('roberta.roberta.embeddings.word_embeddings.learned_embedding'):
        #     #     p.requires_grad = False
        #     print(f'{n} requires grad: {p.requires_grad}')

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        mask_token_indices = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        mask_output = sequence_output[mask_token_indices[0], mask_token_indices[1], :]
        mask_scores = self.lm_head(self.mul_grad(mask_output))
        label_word_scores = torch.stack([mask_scores[i, self.label_ids] for i in range(len(mask_token_indices[0]))])
        
        masked_lm_loss = None
        # if labels is not None:
            # loss_fct = nn.BCEWithLogitsLoss()
            # masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            # masked_lm_loss = loss_fct(label_word_scores, F.one_hot(labels, num_classes=2).float())
            # loss_fct = nn.CrossEntropyLoss()
            # masked_lm_loss = loss_fct(label_word_scores, labels.view(-1))
            
        discriminator_output = self.discriminator(mask_output) # (batch_size, 1)

        if not return_dict:
            # output = (prediction_scores,) + outputs[2:]
            output = (label_word_scores, discriminator_output) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=label_word_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), discriminator_output

            
class RoBERTa_AdSPT_no_verb(RobertaPreTrainedModel):

    def __init__(self, pretrained_model_name_or_path, config, args, initialize_from_vocab=True, token4head='mask', num_class=2, loss_coeff=None):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        # do not change name of self.roberta
        self.roberta = AutoModel.from_pretrained(pretrained_model_name_or_path, add_pooling_layer=False)
        self.args = args
        # domain discriminator
        self.discriminator = AdversarialNetwork2(self.roberta.config.hidden_size)
        # default loss coeff: set learning rate of L_class on PLM same as lr on classifier
        if loss_coeff is None:
            loss_coeff = self.args.train.lr/self.args.train.plm_lr
        # to implement update with L_class on classifier and (lambda * L_class) on PLM
        # multiply grad before classifier
        self.mul_grad = GradientReverseModule(lambda step: -loss_coeff) # lambda
        self.token4head = token4head.lower() # choice: [mask, cls]
        self.num_class = num_class
        self.classifier = CLS(self.roberta.config.hidden_size, self.num_class, bottle_neck_dim=256)
        
        # Initialize weights and apply final processing
        s_wte = SoftEmbedding(self.roberta.get_input_embeddings(), 
                            mask_token_id=self.tokenizer.mask_token_id,
                            n_tokens=args.train.n_tokens, 
                            initialize_from_vocab=initialize_from_vocab)
        self.set_input_embeddings(s_wte)
        print(f'Initialized adspt with no verbalizer on a single source domain')

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0] # last_hidden_states
        mask_token_indices = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True) # logits on mask
        assert self.token4head in ['mask', 'cls']
        if self.token4head == 'mask':  # use mask token on classifier
            state = sequence_output[mask_token_indices[0], mask_token_indices[1], :]
        else:  # use cls token on classifier
            state = sequence_output[:, 0, :]
        
        masked_lm_loss = None
        logits = self.classifier(self.mul_grad(state))[2] # classifier logits
        
        discriminator_output = self.discriminator(state) # discriminator logits, (batch_size, 1)

        if not return_dict:
            output = (logits, discriminator_output) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        return MaskedLMOutput(
            loss=masked_lm_loss,
            # logits=prediction_scores,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), discriminator_output
               
class BERT_AdSPT_no_verb(BertPreTrainedModel):

    def __init__(self, pretrained_model_name_or_path, config, args, initialize_from_vocab=True, token4head='mask', num_class=2, loss_coeff=None):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        # do not change name of self.bert
        # change if error related to self.base_model occurs
        self.bert = AutoModel.from_pretrained(pretrained_model_name_or_path, add_pooling_layer=False)
        self.args = args
        self.discriminator = AdversarialNetwork2(self.bert.config.hidden_size)
        if loss_coeff is None:
            loss_coeff = self.args.train.lr/self.args.train.plm_lr
        self.mul_grad = GradientReverseModule(lambda step: -loss_coeff)
        self.token4head = token4head.lower() # choice: [mask, cls]
        self.num_class = num_class
        self.classifier = CLS(self.bert.config.hidden_size, self.num_class, bottle_neck_dim=256)
        # Initialize weights and apply final processing
        s_wte = SoftEmbedding(self.bert.get_input_embeddings(), 
                            mask_token_id=self.tokenizer.mask_token_id,
                            n_tokens=args.train.n_tokens, 
                            initialize_from_vocab=initialize_from_vocab)
        self.set_input_embeddings(s_wte)
        print(f'Initialized adspt with no verbalizer on a single source domain')

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        mask_token_indices = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        assert self.token4head in ['mask', 'cls']
        if self.token4head == 'mask':  
            state = sequence_output[mask_token_indices[0], mask_token_indices[1], :]
        else:
            state = sequence_output[:, 0, :]
        
        masked_lm_loss = None
        # breakpoint()
        logits = self.classifier(self.mul_grad(state))[2]
        
        discriminator_output = self.discriminator(state) # (batch_size, 1)

        if not return_dict:
            output = (logits, discriminator_output) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        return MaskedLMOutput(
            loss=masked_lm_loss,
            # logits=prediction_scores,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), discriminator_output

    
if __name__ == '__main__':
    config = AutoConfig.from_pretrained('roberta-base', cache_dir='/home/pch330/data/model_data')
    model = RoBERTa_AdSPT_mlm_only(
        'roberta-base',
        config=config,
        n_tokens=3,
        initialize_from_vocab=True
    )