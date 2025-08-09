import torch
import torch.nn as nn
import math
class WordEmbedd(nn.Module):
  def __init__(self,d_model:int,vocab_size:int):
    super().__init__()
    self.d_model=d_model
    self.vocab_size=vocab_size
    self.embedding=nn.Embedding(vocab_size,d_model)
  def forward(self,x):
    return self.embedding(x)*math.sqrt(self.d_model)

class positionalEncodding(nn.Module):
  def __init__(self,d_model:int,seq_length:int,dropout:int):
    super().__init__()
    self.d_model=d_model
    self.seq_length=seq_length
    self.dropout=nn.Dropout(dropout)
    #creating a matrix of shape (seq_length, d_model)
    pe=torch.zeros(seq_length,d_model)
    #create a vector of shape(seq_lenth)
    pos=torch.arange(0,seq_length,dtype=torch.float).unsqueeze(1)
    div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
    #[:,0::2] first : was to select all rows;second 0::2 means we start from 0 th column and step=2
    pe[:,0::2]=torch.sin(pos*div_term)
    pe[:,1::2]=torch.cos(pos*div_term)

    pe=pe.unsqueeze(0) #added a new dimension to pe for batch
    self.register_buffer('pe',pe)
  def forward(self,x):
    #apply pe to every word
    x=x+(self.pe[:,:x.shape[1],:]).requires_grad_(False) #because pe is fixed, not learnable
    return self.dropout(x)
class Layer_Norm(nn.Module):
  def __init__(self,eps:float=10**-6):
    super().__init__()
    self.eps=eps
    self.alpha=nn.Parameter(torch.ones(1))
    self.beta=nn.Parameter(torch.ones(1))
  def forward(self,x):
    mean=x.mean(dim=-1,keepdim=True)
    std=x.std(dim=-1,keepdim=True)
    return self.alpha*(x-mean)/torch.sqrt(std**2+self.eps) +self.beta
class Feed_Forward(nn.Module):
  def __init__(self,d_ff:int,d_model:int,dropout:float):
    super().__init__()
    self.linear1=nn.Linear(d_model,d_ff)
    self.dropout=nn.Dropout(dropout)
    self.linear2=nn.Linear(d_ff,d_model)
  def forward(self,x):
    # input (batch,seq_len,d_model) --(linear1)-->(batch,seq_len,d_ff)--(linear2)-->(batch,seq_len,d_model)
    #FFN=max(0,xW1+b1)w2 +b2
    return self.linear2(self.dropout(torch.relu(self.linear1(x))))
class MultiHeadAttention(nn.Module):
  def __init__(self,d_model:int,seq_len:int,dropout:float,h:int):
    super().__init__()
    self.h=h
    self.d_model=d_model
    assert d_model%h==0,"d_model not divisible by h"
    self.d_k=d_model//h
    self.w_k=nn.Linear(d_model,d_model)
    self.w_q=nn.Linear(d_model,d_model)
    self.w_v=nn.Linear(d_model,d_model)
    self.w_o=nn.Linear(d_model,d_model)
    self.dropout=nn.Dropout(dropout)
  #can call the attention function without having any instance of the class like,
  # MultiHeadAttention.attention
  @staticmethod
  def attention(query,key,value,mask,dropout:float):
    d_k=query.shape[-1]
    #query(batch,h,seq_len,d_k) kets.T(batch,h,d_k,seq_len)-->(batch,h,seq_len,seq_len)
    attention_scores=(query@key.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
      attention_scores.masked_fill_(mask==0,-1e9)
    attention_scores=attention_scores.softmax(dim=-1)#(batch,h,seq_len,seq_len)
    if dropout is not None:
      attention_scores=dropout(attention_scores)
    return(attention_scores@ value),attention_scores #(batch,h,seq_len,d_k)
  def forward(self,q,k,v,mask):
    query=self.w_q(q)
    key=self.w_k(k)
    value=self.w_v(v)
    #query.dim-->(batch,seq,d_model)--->(batch,seq_len,h,d_k)-->(batch,h,seq_len,d_k)
    #view is to reshape a tensor without copying data like numpy's reshape()
    query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
    key=key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
    value=value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)
    x,self.attention_scores=MultiHeadAttention.attention(query,key,value,mask,self.dropout)
    #(batch,h,seq_len,d_k)-->(batch,seq_len,h,d_k)-->(batch,seq_len,d_model)
    x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)
    return self.w_o(x)
class Residual_Connections(nn.Module):
  def __init__(self,features:int,dropout:float) -> None:
    super().__init__()
    self.dropout=nn.Dropout(dropout)
    self.norm=Layer_Norm(features)
  def forward(self,x,sublayer):
    return x+ self.dropout(sublayer(self.norm(x)))
class EncoderBlock(nn.Module):
  def __init__(self,features:int,attenton_block:MultiHeadAttention,dropout:float,forward:Feed_Forward):
    super().__init__()
    self.attention_block=attenton_block
    self.feed_forward=forward
    self.residual_network=nn.ModuleList([Residual_Connections(features,dropout) for i in range(2)])

  def forward(self,x,src_mask):
    x=self.residual_network[0](x,lambda x:self.attention_block(x,x,x,src_mask))
    x=self.residual_network[1](x,lambda x:self.feed_forward(x))
    return x
class Encoder(nn.Module):
  def __init__(self,features:int,layers:nn.ModuleList)-> None:
    super().__init__()
    self.layers=layers
    self.norm=Layer_Norm(features)
  def forward(self,x,mask):
    for layer in self.layers:
      x=layer(x,mask)
    return self.norm(x)

class DecoderBlock(nn.Module):
  def __init__(self,features:int,self_attention:MultiHeadAttention,cross_attention:MultiHeadAttention,feed_forward:Feed_Forward,dropout:float) -> None:
    super().__init__()
    self.self_attention=self_attention
    self.cross_attention=cross_attention
    self.feed_forward=feed_forward
    self.connections=nn.ModuleList([Residual_Connections(features,dropout) for _ in range(3)])
    #src_mask comes from input and tgt_mask form output
  def forward(self,x,encoder_output,src_mask,target_mask):
    x=self.connections[0](x,lambda x:self.self_attention(x,x,x,target_mask))
    x=self.connections[1](x,lambda x:self.cross_attention(x,encoder_output,encoder_output,src_mask))
    x=self.connections[2](x,self.feed_forward)
    return x
class Decoder(nn.Module):
  def __init__(self,features:int,layers:nn.ModuleList):
    super().__init__()
    self.layers=layers
    self.norm=Layer_Norm(features)
  def forward(self,x,encoder_output,src_mask,target_mask):
    for layer in self.layers:
      x=layer(x,encoder_output,src_mask,target_mask)
    return self.norm(x)
#dim of x(batch_size,seq_len,d_model)-----> (batch_size,seq_len,vocab_size)
class LinearProjection(nn.Module):
  def __init__(self,d_model:int,vocab_size:int) -> None:
    super().__init__()
    self.project=nn.Linear(d_model,vocab_size)
  def forward(self,x):
    return torch.log_softmax(self.project(x),dim=-1)
#dim=0: across batches (2 items)
#dim=1: across time steps or sequence positions (4 tokens)
#dim=-1 or dim=2: across vocabulary size (5 possible words)
class Transformer(nn.Module):
  def __init__(self,encoder:Encoder,decoder:Decoder,src_embed: WordEmbedd,tgt_embed:WordEmbedd,src_pos:positionalEncodding,tgt_pos:positionalEncodding,project:LinearProjection)->None:
    super().__init__()
    self.encoder=encoder
    self.decoder=decoder
    self.src_embed=src_embed
    self.tgt_embed=tgt_embed
    self.tgt_pos=tgt_pos
    self.src_pos=src_pos
    self.project_layer=project # Renamed to avoid conflict with method name
  def encode(self,src,src_mask):
    #(batch,seq,d_model)
    src=self.src_embed(src)
    src=self.src_pos(src)
    return self.encoder(src,src_mask)
  def decode(self,encoder_output:torch.Tensor,src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
    tgt=self.tgt_embed(tgt)
    tgt=self.tgt_pos(tgt)
    return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
  def project(self,x):
    return self.project_layer(x) # Use the renamed project layer
    
def build_transformer(src_vocab_size:int,tgt_vocab_size:int,src_seq_len:int,tgt_seq_len:int,d_model:int=512,N:int=6,h: int=8,dropout:float=0.1,d_ff:int=2048):
    src_embed=WordEmbedd(d_model,src_vocab_size)
    tgt_embed=WordEmbedd(d_model,tgt_vocab_size)


    src_pos=positionalEncodding(d_model,src_seq_len,dropout)
    tgt_pos=positionalEncodding(d_model,tgt_seq_len,dropout)

    encoder_blocks=[]
    for _ in range(N):
      encoder_self_attention=MultiHeadAttention(d_model,src_seq_len,dropout,h)
      feed_forward_block=Feed_Forward(d_ff,d_model,dropout)
      encoder_block=EncoderBlock(d_model,encoder_self_attention,dropout,feed_forward_block)
      encoder_blocks.append(encoder_block)
    decoder_blocks=[]
    for _ in range(N):
      decoder_self_attention=MultiHeadAttention(d_model,tgt_seq_len,dropout,h)
      decoder_cross_attention=MultiHeadAttention(d_model,tgt_seq_len,dropout,h)
      feed_forward_block=Feed_Forward(d_ff,d_model,dropout)
      decoder_block=DecoderBlock(d_model,decoder_self_attention,decoder_cross_attention,feed_forward_block,dropout)
      decoder_blocks.append(decoder_block)

    encoder=Encoder(d_model,nn.ModuleList(encoder_blocks))
    decoder=Decoder(d_model,nn.ModuleList(decoder_blocks))

    projection_layer=LinearProjection(d_model,tgt_vocab_size)

    transformer=Transformer(encoder,decoder,src_embed,tgt_embed,src_pos,tgt_pos,projection_layer)

    for p in transformer.parameters():
      if p.dim()>1:
        nn.init.xavier_uniform_(p)

    return transformer
