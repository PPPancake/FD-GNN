import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import random

class MLP_(nn.Module):
	def __init__(self, features, feature_dim, embed_dim, cuda=True):
		super(MLP_, self).__init__()

		self.features = features
		self.feature_dim = feature_dim
		self.embed_dim = embed_dim
		self.cuda = cuda
		self.mlp_layer = nn.Linear(self.feature_dim, self.embed_dim)
	
	def forward(self, nodes):
		if(self.cuda):
			batch_features = self.features(torch.cuda.LongTensor(nodes))
		else:
			batch_features = self.features(torch.LongTensor(nodes))
		
		if self.cuda:
			self.mlp_layer.cuda()
		
		result = self.mlp_layer(batch_features)
		result = F.relu(result)
		return result

def weight_inter_agg(num_relations, neigh_feats, embed_dim, alpha, n, cuda): # 实现基于权重的多关系邻居聚合
	"""
	Weight inter-relation aggregator
	:param num_relations: number of relations in the graph
	:param neigh_feats: intra_relation aggregated neighbor embeddings for each aggregation
	:param embed_dim: the dimension of output embedding
	:param alpha: weight paramter for each relation
	:param n: number of nodes in a batch
	:param cuda: whether use GPU
	"""
	neigh_h = neigh_feats.t()
	w = F.softmax(alpha, dim = 1)
	if cuda:
		aggregated = torch.zeros(size=(embed_dim, n)).cuda()
	else:
		aggregated = torch.zeros(size=(embed_dim, n))
	result = []
	for r in range(num_relations):
		tmp = torch.mul(w[:, r].unsqueeze(1).repeat(1,n), neigh_h[:, r*n:(r+1)*n])
		#aggregated += tmp
		#print(tmp.shape[0], tmp.shape[1])
		result.append(tmp.t())
	#return aggregated.t()
	return result

class InterAgg(nn.Module):
	def __init__(self, features, feature_dim, embed_dim, 
				 train_pos, train_neg, adj_lists, intraggs, inter='GNN', cuda=True):
		"""
		Initialize the inter-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feature_dim: the input dimension
		:param embed_dim: the embed dimension
		:param train_pos: positive samples in training set
		:param train_neg: negative samples in training set
		:param adj_lists: a list of adjacency lists for each single-relation graph
		:param intraggs: the intra-relation aggregators used by each single-relation graph
		:param inter: NOT used in this version, the aggregator type: 'Att', 'Weight', 'Mean', 'GNN'
		:param cuda: whether to use GPU
		"""
		super(InterAgg, self).__init__()

		self.features = features
		self.pos_vector = None
		self.neg_vector = None

		self.softmax = nn.Softmax(dim=-1)
		self.KLDiv = nn.KLDivLoss(reduction='batchmean')
		self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

		self.dropout = 0.6
		self.adj_lists = adj_lists
		self.intra_agg1 = intraggs[0]
		self.intra_agg2 = intraggs[1]
		self.intra_agg3 = intraggs[2]
		self.embed_dim = embed_dim
		self.feat_dim = feature_dim
		self.inter = inter
		self.cuda = cuda
		self.intra_agg1.cuda = cuda
		self.intra_agg2.cuda = cuda
		self.intra_agg3.cuda = cuda
		self.train_pos = train_pos
		self.train_neg = train_neg
		
		# initial filtering thresholds
		self.thresholds = [0.5, 0.5, 0.5]

		# parameter used to transform node embeddings before inter-relation aggregation
		#self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim*len(intraggs)+self.feat_dim, self.embed_dim))
		self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim+self.feat_dim, self.embed_dim))
		init.xavier_uniform_(self.weight)

		# label predictor for similarity measure
		self.label_clf = nn.Linear(self.feat_dim, 2)

		# initialize the parameter logs
		self.weights_log = []
		self.thresholds_log = [self.thresholds]
		self.relation_score_log = []

		if self.cuda and isinstance(self.train_pos, list) and isinstance(self.train_neg, list):
			self.pos_index = torch.LongTensor(self.train_pos).cuda()
			self.neg_index = torch.LongTensor(self.train_neg).cuda()
		else:
			self.pos_index = torch.LongTensor(self.train_pos)
			self.neg_index = torch.LongTensor(self.train_neg)
		
		# 初始化用于计算注意力权重的参数aplha
		if self.cuda:
			self.alpha = nn.Parameter(torch.FloatTensor(self.embed_dim, 3)).cuda()

		else:
			self.alpha = nn.Parameter(torch.FloatTensor(self.embed_dim, 3))

		init.xavier_uniform_(self.alpha)
	
	def forward(self, nodes):
		"""
		:param nodes: a list of batch node ids
		"""

		# extract 1-hop neighbor ids from adj lists of each single-relation graph
		to_neighs = []
		for adj_list in self.adj_lists:
			to_neighs.append([set(adj_list[int(node)]) for node in nodes])
		
		# get neighbor node id list for each batch node and relation
		r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
		r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
		r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]

		# find unique nodes and their neighbors used in current batch
		unique_nodes = set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]),
									set.union(*to_neighs[2], set(nodes)))
		self.unique_nodes = unique_nodes

		if self.cuda and isinstance(nodes, list):
		#	batch_features = self.mlp(torch.cuda.LongTensor(list(unique_nodes)))
			self_feats = self.features(torch.LongTensor(nodes).cuda())
		else:
		#	batch_features = self.mlp(torch.LongTensor(list(unique_nodes)))
			self_feats = self.features(torch.LongTensor(nodes))

		r1_feats = self.intra_agg1.forward(nodes, r1_list, self.features)
		r2_feats = self.intra_agg2.forward(nodes, r2_list, self.features)
		r3_feats = self.intra_agg3.forward(nodes, r3_list, self.features)
		
		#self_feats = self.fetch_feat(nodes)

		# Update label vector
		self.update_label_vector(self.features)

		# concat the intra-aggregated embeddings from each relatio
		#neigh_feats = torch.cat((r1_feats, r2_feats, r3_feats), dim = 0)
		#attention_layer_outputs = weight_inter_agg(len(self.adj_lists), neigh_feats, self.embed_dim, self.alpha, len(nodes), self.cuda)
		#r1_feats = attention_layer_outputs[0]
		#r2_feats = attention_layer_outputs[1]
		#r3_feats = attention_layer_outputs[2]

		#cat_feats = torch.cat((self_feats, attention_layer_outputs), dim=1)
		cat_feats = torch.cat((self_feats, r1_feats, r2_feats, r3_feats), dim=1)

		return cat_feats
	
	def softmax_with_temperature(self, input, t=1, axis=-1):
		ex = torch.exp(input/t)
		sum = torch.sum(ex, axis=axis)
		return ex/sum
	
	def update_label_vector(self, x):
		if isinstance(x, torch.Tensor):
			x_pos = x[self.train_pos]
			x_neg = x[self.train_neg]
		elif isinstance(x, torch.nn.Embedding):
			x_pos = x(self.pos_index)
			x_neg = x(self.neg_index)
		if self.pos_vector is None:
			self.pos_vector = torch.mean(x_pos, dim=0, keepdim=True).detach() # proto+
			self.neg_vector = torch.mean(x_neg, dim=0, keepdim=True).detach() # proto-
		else:
			cosine_pos = self.cos(self.pos_vector, x_pos)
			cosine_neg = self.cos(self.neg_vector, x_neg)
			weights_pos = self.softmax_with_temperature(cosine_pos, t=5).reshape(1, -1)
			weights_neg = self.softmax_with_temperature(cosine_neg, t=5).reshape(1, -1)
			self.pos_vector = torch.mm(weights_pos, x_pos).detach()
			self.neg_vector = torch.mm(weights_neg, x_neg).detach()
	
	def fl_loss(self, grads_idx):
		x = F.log_softmax(self.features(self.pos_index)[:, grads_idx], dim=1)
		target_pos = self.pos_vector[:, grads_idx].repeat(x.shape[0], 1).softmax(dim=-1) # proto+
		target_neg = self.neg_vector[:, grads_idx].repeat(x.shape[0], 1).softmax(dim=-1) # proto-
		loss_pos = self.KLDiv(x, target_pos)
		loss_neg = self.KLDiv(x, target_neg)
		return loss_pos, loss_neg
	
	def fn_loss(self, nodes, non_grad_idx):
		pos_nodes = set(self.train_pos)
		to_neighs = []
		target = []
		for adj_list in self.adj_lists:
			target_r = []
			to_neighs_r = []
			for node in nodes:
				if int(node) in pos_nodes:
					target_r.append(int(node))
					to_neighs_r.append(set(adj_list[int(node)]))
			to_neighs.append(to_neighs_r)
			target.append(target_r)
		
		to_neighs_all = []
		for x, y, z in zip(to_neighs[0], to_neighs[1], to_neighs[2]):
			to_neighs_all.append(set.union(x, y, z))
		
		r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
		r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
		r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]
		
		pos_1, neg_1 = self.intra_agg1.fn_loss(non_grad_idx, target[0], r1_list, self.unique_nodes, to_neighs_all, self.features)
		pos_2, neg_2 = self.intra_agg2.fn_loss(non_grad_idx, target[1], r2_list, self.unique_nodes, to_neighs_all, self.features)
		pos_3, neg_3 = self.intra_agg3.fn_loss(non_grad_idx, target[2], r3_list, self.unique_nodes, to_neighs_all, self.features)
		return pos_1 + pos_2 + pos_3, neg_1 + neg_2 + neg_3

class IntraAgg(nn.Module):
	def __init__(self, feat_dim, embed_dim, cuda=False):
		"""
		Initialize the intra-relation aggregator
		:param feat_dim: the input dimension
		:param embed_dim: the embed dimension
		:param cuda: whether to use GPU
		"""
		super(IntraAgg, self).__init__()

		self.embed_dim = embed_dim
		self.feat_dim = feat_dim
		self.cuda = cuda
		
		self.weight = nn.Parameter(torch.FloatTensor(2*self.feat_dim, self.embed_dim))
		init.xavier_uniform_(self.weight)
		self.KLDiv = nn.KLDivLoss(reduction='batchmean')
	
	def forward(self, nodes, to_neighs_list, features):
		"""
		Code partially from https://github.com/williamleif/graphsage-simple/
		:param nodes: list of nodes in a batch
		:param to_neighs_list: neighbor node id list for each batch node in one relation
		:param features: the input node features or embeddings for all nodes
		"""
		samp_neighs = [set(x) for x in to_neighs_list]
		unique_nodes_list = list(set.union(*samp_neighs))
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
		#neighbors_new_index = [unique_nodes[n] for n in unique_nodes_list]

		# 表示每个节点在整个特征矩阵中的位置
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1 # 对值为1的节点进行采样
		if self.cuda:
			mask = mask.cuda()
		num_neigh = mask.sum(1, keepdim=True)
		mask = mask.div(num_neigh)# 归一化的权重矩阵
		
		if self.cuda:
			self_feats = features(torch.LongTensor(nodes).cuda()) # 当前批次节点的特征
			embed_matrix = features(torch.LongTensor(unique_nodes_list).cuda()) # 唯一邻居节点的特征
		else:
			self_feats = features(torch.LongTensor(nodes))
			embed_matrix = features(torch.LongTensor(unique_nodes_list))

		agg_feats = mask.mm(embed_matrix) # 得到v的所有邻居节点的加权平均特征
		diff_feats = self_feats - agg_feats
		cat_feats = torch.cat((diff_feats, agg_feats), dim=1) # 自身特征与邻居特征进行聚合
		to_feats = F.relu(cat_feats.mm(self.weight))

		return to_feats

	def fetch_feat(self, features, nodes):
		if self.cuda and isinstance(nodes, list):
			index = torch.LongTensor(nodes).cuda()
		else:
			index = torch.LongTensor(nodes)
		return features(index)
	
	def fn_loss(self, non_grad_idx, target, neights, all_nodes, all_neighs, features):
		x = F.log_softmax(self.fetch_feat(features, target)[:, non_grad_idx], dim=-1)
		pos = torch.zeros_like(self.fetch_feat(features, target))
		neg = torch.zeros_like(self.fetch_feat(features, target))
		for i in range(len(target)):
			pos[i] = torch.mean(self.fetch_feat(features, neights[i]), dim=0, keepdim=True)
			neg_idx = [random.choice(list(all_nodes.difference(all_neighs[i])))]
			neg[i] = self.fetch_feat(features, neg_idx)
		pos = pos[:, non_grad_idx].softmax(dim=-1)
		neg = neg[:, non_grad_idx].softmax(dim=-1)
		loss_pos = self.KLDiv(x, pos) # KL(S_u, S-v), u邻居节点们
		loss_neg = self.KLDiv(x, neg) # KL(S_u, S-v), u非邻居节点
		return loss_pos, loss_neg