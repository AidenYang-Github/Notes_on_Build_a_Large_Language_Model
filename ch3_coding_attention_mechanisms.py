import torch
import torch.nn as nn
from numpy.ma.core import diagonal

# === 输入词元 ==============================
# 输入向量（每一行代表一个词元，分别是'Your'/'journey'/'starts'/'with'/'one'/'step'）
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]]
)
"""
inputs.shape
torch.Size([6, 3])
"""

# === 注意力分数 ==============================
# 设置查询向量（取输入向量的第二个作为查询向量）
query = inputs[1]

# 计算注意力分数
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)
"""
tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])"""

# === 注意力权重 ==============================
# 计算注意力权重（简单归一化）
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print(f"Attention weights:{attn_weights_2_tmp}")
print(f"Sum:{attn_weights_2_tmp.sum()}")
"""
Attention weights:tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
Sum:1.0000001192092896"""


# 计算注意力权重（softmax归一化）
# 使用torch.exp()构建softmax函数
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


attn_weights_2_naive = softmax_naive(attn_scores_2)
print(f"Attention weights:{attn_weights_2_naive}")
print(f"Sum:{attn_weights_2_naive.sum()}")
"""
Attention weights:tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
Sum:1.0"""

# 直接使用torch.softmax()函数
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print(f"Attention weights:{attn_weights_2}")
print(f"Sum:{attn_weights_2.sum()}")
"""
Attention weights:tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
Sum:1.0"""

# === 上下文向量 ==============================
# 通过第二个词元的注意力权重向量计算第二个词元的上下文向量
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print(f"context_vec_2:{context_vec_2}")
"""
context_vec_2:tensor([0.4419, 0.6515, 0.5683])
"""

# 计算所有输入向量的注意力分数，并计算注意力权重，然后再计算所有输入向量的上下文向量
# 我的方法
attn_scores = torch.empty(inputs.shape[0], inputs.shape[0])
for j in range(len(inputs)):
    query = inputs[j]
    attn_scores_j = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_j[i] = torch.dot(inputs[i], query)
    attn_scores[j] = attn_scores_j
print(f"attn_scores:{attn_scores}")
"""
attn_scores:tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
        [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
        [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
        [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
        [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
        [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])"""

# 书中的方法(将输入向量中逐向量相乘)
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)
"""
tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
        [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
        [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
        [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
        [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
        [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])"""

# 书中的方法（矩阵相乘）
attn_scores = inputs @ inputs.T
print(attn_scores)
"""
tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
        [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
        [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
        [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
        [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
        [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])"""

# 使用softmax进行注意力分数归一化
"""
dim 参数用于指定输入张量的计算维度。将 dim 设置为-1 表示让 softmax 函数在 attn_scores 张量的最后一个维度上进行归一化
"""
attn_weights = torch.softmax(attn_scores, dim=-1)  # 以最后一个维度（列）的数据进行归一化
print(f"attn_weights:{attn_weights}")
"""
attn_weights:tensor([[0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452],
        [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581],
        [0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565],
        [0.1435, 0.2074, 0.2046, 0.1462, 0.1263, 0.1720],
        [0.1526, 0.1958, 0.1975, 0.1367, 0.1879, 0.1295],
        [0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]])"""

# 计算第二行向量的注意力权重的和
row_2_sum = sum(attn_weights[1])
print(f"Row 2 sum:{row_2_sum}")
print(f"All row sums:{attn_weights.sum(dim=-1)}")
"""
Row 2 sum:1.0
All row sums:tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
"""

# 通过注意力权重矩阵和输入向量矩阵通过矩阵乘法计算所有上下文向量
all_context_vecs = attn_weights @ inputs
print(f"all_context_vecs:{all_context_vecs}")
"""
all_context_vecs:tensor([[0.4421, 0.5931, 0.5790],
        [0.4419, 0.6515, 0.5683],
        [0.4431, 0.6496, 0.5671],
        [0.4304, 0.6298, 0.5510],
        [0.4671, 0.5910, 0.5266],
        [0.4177, 0.6503, 0.5645]])
all_context_vecs.shape
PyDev console: starting.
torch.Size([6, 3])"""

# === 3.4 实现带可训练权重的自注意力机制 ==============================
"""
初始化权重矩阵——>
"""
"""请注意，在类 GPT 模型中，输入和输出的维度通常是相同的，但为了便于理解计算过程，这里
我们使用不同的输入维度（d_in=3）和输出维度（d_out=2）"""
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

# 初始化3个权重矩阵Wq、Wk和Wv
"""
设置 requires_grad=False 以减少输出中的其他项，但如果要在模型训练中使用这些权重矩
阵，就需要设置 requires_grad=True，以便在训练中更新这些矩阵"""
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# 计算查询向量、键向量和值向量
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(f"query_2:{query_2}")
print(f"key_2:{key_2}")
print(f"value_2:{value_2}")
"""
query_2:tensor([0.5869, 1.3484])
key_2:tensor([0.6757, 1.0007])
value_2:tensor([1.2966, 0.4479])
"""

# 通过矩阵乘法得到所有键向量和值向量
queries = inputs @ W_query
keys = inputs @ W_key
values = inputs @ W_value
print(f"queries:{queries}")
print(f"keys:{keys}")
print(f"values:{values}")
print(f"keys.shape:{keys.shape}")
print(f"values.shape:{values.shape}")
"""
keys:tensor([[0.2932, 0.5969],
        [0.8787, 0.9615],
        [0.8602, 0.9475],
        [0.5585, 0.5491],
        [0.2833, 0.4289],
        [0.7664, 0.7184]])
values:tensor([[1.1687, 1.2688],
        [1.7188, 1.2836],
        [1.6915, 1.2801],
        [0.9721, 0.6145],
        [0.7222, 0.8569],
        [1.2986, 0.6999]])
keys.shape:torch.Size([6, 2])
values.shape:torch.Size([6, 2])"""

# 计算注意力分数w22
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(f"attn_score_22:{attn_score_22}")
"""
未归一化的注意力分数
attn_score_22:2.002469778060913
"""

# 通过矩阵乘法将这个计算推广到所有的注意力分数
attn_scores_2 = query_2 @ keys.T  # 给定query的全部注意力分数
print(f"attn_scores_2:{attn_scores_2}")
"""
tensor([1.4159, 2.4447, 2.4285, 1.3567, 1.4547, 1.6005])"""

# 将注意力分数转换为注意力权重
"""
（通过缩放注意力分数并应用softmax函数来计算注意力权重）
缩放方式：通过将注意力分数除以键向量的嵌入维度的平方根来进行缩放（取平方根在数学上等同于以0.5为指数进行幂运算）
"""
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)
print(f"attn_weights_2:{attn_weights_2}")
"""
attn_weights_2:tensor([0.1476, 0.2164, 0.2134, 0.1365, 0.1240, 0.1621])"""

# 通过对值向量进行加权求和来计算上下文向量（注意力权重作为加权因子，用于权衡每个值向量的重要性）
context_vec_2 = attn_weights_2 @ values
print(f"context_vec_2:{context_vec_2}")
"""
context_vec_2:tensor([0.8169, 1.0224])"""

# 计算输入序列中z(1)到z(T)的所有上下文向量
attn_scores = queries @ keys.T

attn_weights = torch.softmax(attn_scores / d_k ** 0.5, dim=-1)

context_vecs = attn_weights @ values


# === 3.4.2 实现一个简化的自注意Python类 ==============================
class SelfAttention_v1(nn.Module):
    """一个简化的自注意力类"""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    """使用PyTorch线性层的自注意力类"""
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


class CausalAttention(nn.Module):
    """因果注意力"""

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))    # register_buffer （注册缓存）调用也是一个新版本（下文提供了更多信息）
        """虽然此时所有新增的代码行都应该是熟悉的，但我们在__init__方法中增加了一个 self.register_buffer()调用。虽然在 PyTorch 中使用 register_buffer 并非所有情况下都是必需的，但在这里具有一些优势。例如，当我们在大语言模型中使用 CausalAttention 类时，缓冲区会与模型一起自动移动到适当的设备（CPU 或 GPU），这在训练大语言模型时非常重要。这意味着我们无须手动确保这些张量与模型参数在同一设备上，从而避免了设备不匹配的错误。"""

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    """    一个实现多头注意力的封装类    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    """    3.6.2 通过权重划分实现多头注意力    """

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 减少投影维度以匹配所需的输出维度

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)  # 使用一个线性层来组合头的输出
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # 张量形状：(b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 通过添加一个num_heads维度来隐式地分隔矩阵。
        # 然后展开最后一个维度：(b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 从形状(b, num_tokens, num_heads, head_dim)转换到(b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)  # 计算每个头的点积
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # 被截断为词元数量的掩码

        attn_scores.masked_fill_(mask_bool, -torch.inf)  # 使用掩码来填充注意力分数

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 张量形状：(b, num_tokens, n_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # 组合头，其中self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # 添加一个可选的线性投影
        context_vec = self.out_proj(context_vec)
        return context_vec


if __name__ == '__main__':
    # === 3.4 实现带可训练权重的自注意力机制 ==============================
    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print(sa_v1(inputs))
    """
    tensor([[0.2996, 0.8053],
        [0.3061, 0.8210],
        [0.3058, 0.8203],
        [0.2948, 0.7939],
        [0.2927, 0.7891],
        [0.2990, 0.8040]], grad_fn=<MmBackward0>)
    """

    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs))
    """
    tensor([[-0.0739,  0.0713],
        [-0.0748,  0.0703],
        [-0.0749,  0.0702],
        [-0.0760,  0.0685],
        [-0.0763,  0.0679],
        [-0.0754,  0.0693]], grad_fn=<MmBackward0>)
    """
    # === 3.5 利用因果注意力隐藏未来词汇 ==============================
    # === 3.5.1 因果注意力的掩码实现 ==============================
    """
    复用3.4.2节中的SelfAttention_v2对象的查询权重矩阵和键权重矩阵"""
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
    print(f"attn_weights:{attn_weights}")
    """
    attn_weights:tensor([[0.1921, 0.1646, 0.1652, 0.1550, 0.1721, 0.1510],
        [0.2041, 0.1659, 0.1662, 0.1496, 0.1665, 0.1477],
        [0.2036, 0.1659, 0.1662, 0.1498, 0.1664, 0.1480],
        [0.1869, 0.1667, 0.1668, 0.1571, 0.1661, 0.1564],
        [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.1585],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<SoftmaxBackward0>)
    """

    # 使用PyTorch的tril函数来创建一个对角线以上元素为0的掩码
    context_length = attn_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print(f"mask_simple:{mask_simple}")
    """掩码矩阵
    mask_simple:tensor(
       [[1., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1.]])
    """
    # 将掩码矩阵和注意力权重矩阵相乘，使对角线上方的值变为0
    masked_simple = attn_weights * mask_simple
    print(f"masked_simple:{masked_simple}")
    """注：此时的掩码注意力权重矩阵已经变成未归一化的矩阵了
    masked_simple:tensor(
       [[0.1921, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2041, 0.1659, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2036, 0.1659, 0.1662, 0.0000, 0.0000, 0.0000],
        [0.1869, 0.1667, 0.1668, 0.1571, 0.0000, 0.0000],
        [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.0000],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<MulBackward0>)
    """

    # 重新归一化注意力权重，使每一行的总和再次为1，通过将每行中的每个元素除以每行中的和来实现
    row_sums = masked_simple.sum(dim=-1, keepdim=True)
    masked_simple_norm = masked_simple / row_sums
    print(f"masked_simple_norm:{masked_simple_norm}")
    """
    masked_simple_norm:tensor(
       [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
        [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
        [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<DivBackward0>)
    """

    # 通过创建一个对角线以上是1的掩码，并将这些1替换为负无穷大（-inf）值，来实现这种更高效的掩码“方法”
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    print(f"masked:{masked}")
    """
    masked:tensor(
       [[0.2899,   -inf,   -inf,   -inf,   -inf,   -inf],
        [0.4656, 0.1723,   -inf,   -inf,   -inf,   -inf],
        [0.4594, 0.1703, 0.1731,   -inf,   -inf,   -inf],
        [0.2642, 0.1024, 0.1036, 0.0186,   -inf,   -inf],
        [0.2183, 0.0874, 0.0882, 0.0177, 0.0786,   -inf],
        [0.3408, 0.1270, 0.1290, 0.0198, 0.1290, 0.0078]],
       grad_fn=<MaskedFillBackward0>)
    """

    # 对这些掩码结果应用softmax函数
    attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
    print(f"attn_weights:{attn_weights}")
    """
    attn_weights:tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
        [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
        [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
        [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
       grad_fn=<SoftmaxBackward0>)"""

    # 计算上下文向量
    context_vecs = attn_weights @ values
    print(f"context_vecs:{context_vecs}")
    """
    tensor([[0.7203, 0.7929],
        [0.8556, 1.1288],
        [0.9083, 1.2557],
        [0.8231, 1.1758],
        [0.8153, 1.1421],
        [0.7724, 1.1131]], grad_fn=<MmBackward0>)"""

    # 3.5.2 利用dropout掩码额外的注意力权重
    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5)
    example = torch.ones(6, 6)
    print(dropout(example))
    """
    tensor([[2., 2., 0., 2., 2., 0.],
        [0., 0., 0., 2., 0., 2.],
        [2., 2., 2., 2., 0., 2.],
        [0., 2., 2., 0., 0., 2.],
        [0., 2., 0., 2., 0., 2.],
        [0., 2., 2., 2., 2., 0.]])  """

    # 对注意力权重矩阵进行dropout操作
    torch.manual_seed(123)
    print(f"dropout_attn_weights:{dropout(attn_weights)}")
    """
    dropout_attn_weights:tensor([[2.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.6739, 0.6640, 0.6621, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.5178, 0.5152, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.4134, 0.0000, 0.3741, 0.0000, 0.0000],
        [0.0000, 0.3559, 0.3539, 0.3135, 0.2956, 0.0000]],
       grad_fn=<MulBackward0>)"""

    # 3.5.3 实现一个简化的因果注意力类
    # 为简单起见，可以通过复制输入文本示例来模拟批量输入
    # 每个batch有两个输入文本，每个输入文本有6个词元，每个词元的嵌入维度为3
    batch = torch.stack((inputs, inputs), dim=0)
    print(f"batch.shape:{batch.shape}")
    """
    生成一个三维张量，每个batch有两个输入文本，每个输入文本有6个词元，每个词元的嵌入维度为3
    batch.shape:torch.Size([2, 6, 3])
    batch
    tensor([[[0.4300, 0.1500, 0.8900],
             [0.5500, 0.8700, 0.6600],
             [0.5700, 0.8500, 0.6400],
             [0.2200, 0.5800, 0.3300],
             [0.7700, 0.2500, 0.1000],
             [0.0500, 0.8000, 0.5500]],
            [[0.4300, 0.1500, 0.8900],
             [0.5500, 0.8700, 0.6600],
             [0.5700, 0.8500, 0.6400],
             [0.2200, 0.5800, 0.3300],
             [0.7700, 0.2500, 0.1000],
             [0.0500, 0.8000, 0.5500]]])"""

    # 按照之前使用SelfAttention类的方式来使用CausalAttention类：
    torch.manual_seed(123)
    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch)
    print(f"context_vecs:{context_vecs}")
    print(f"context_vecs.shape:{context_vecs.shape}")
    """
    context_vecs:tensor([[[-0.4519,  0.2216],
         [-0.5874,  0.0058],
         [-0.6300, -0.0632],
         [-0.5675, -0.0843],
         [-0.5526, -0.0981],
         [-0.5299, -0.1081]],
        [[-0.4519,  0.2216],
         [-0.5874,  0.0058],
         [-0.6300, -0.0632],
         [-0.5675, -0.0843],
         [-0.5526, -0.0981],
         [-0.5299, -0.1081]]], grad_fn=<UnsafeViewBackward0>)
    context_vecs.shape:torch.Size([2, 6, 2])"""

    # 3.6 将单头注意力扩展到多头注意力
    # 3.6.1 叠加多个单头注意力层
    # 像之前使用CausalAttention类一样使用MultiHeadAttentionWrapper类：
    torch.manual_seed(123)
    context_length = batch.shape[1]  # 这是词元的数量
    d_in, d_out = 3, 2
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)

    print(f"context_vecs:{context_vecs}")
    print(f"context_vecs.shape:{context_vecs.shape}")
    """
    context_vecs:tensor(
       [[[-0.4519,  0.2216,  0.4772,  0.1063],
         [-0.5874,  0.0058,  0.5891,  0.3257],
         [-0.6300, -0.0632,  0.6202,  0.3860],
         [-0.5675, -0.0843,  0.5478,  0.3589],
         [-0.5526, -0.0981,  0.5321,  0.3428],
         [-0.5299, -0.1081,  0.5077,  0.3493]],
        [[-0.4519,  0.2216,  0.4772,  0.1063],
         [-0.5874,  0.0058,  0.5891,  0.3257],
         [-0.6300, -0.0632,  0.6202,  0.3860],
         [-0.5675, -0.0843,  0.5478,  0.3589],
         [-0.5526, -0.0981,  0.5321,  0.3428],
         [-0.5299, -0.1081,  0.5077,  0.3493]]], grad_fn=<CatBackward0>)
    context_vecs.shape:torch.Size([2, 6, 4])"""

    # 3.6.2 通过权重划分实现多头注意力
    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(f"context_vecs:{context_vecs}")
    print(f"context_vecs.shape:{context_vecs.shape}")
    """
    context_vecs:tensor([[[0.3190, 0.4858],
         [0.2943, 0.3897],
         [0.2856, 0.3593],
         [0.2693, 0.3873],
         [0.2639, 0.3928],
         [0.2575, 0.4028]],
        [[0.3190, 0.4858],
         [0.2943, 0.3897],
         [0.2856, 0.3593],
         [0.2693, 0.3873],
         [0.2639, 0.3928],
         [0.2575, 0.4028]]], grad_fn=<ViewBackward0>)
context_vecs.shape:torch.Size([2, 6, 2])
    """

pass
