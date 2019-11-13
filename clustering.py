import time
import random
from math import sqrt


class DisjointSetUnion:
    def __init__(self, size):
        self.parent_ = list(range(size))
        self.ranks_ = [0] * size

    def find(self, node):
        if self.parent_[node] != node:
            self.parent_[node] = self.find(self.parent_[node])
        return self.parent_[node]

    def union_sets(self, first, second):
        f_root = self.find(first)
        s_root = self.find(second)
        if f_root == s_root:
            return False
        if self.ranks_[f_root] < self.ranks_[s_root]:
            self.parent_[f_root] = s_root
        elif self.ranks_[f_root] > self.ranks_[s_root]:
            self.parent_[s_root] = f_root
        else:
            self.parent_[s_root] = f_root
            self.ranks_[f_root] += 1
        return True

    def result(self):
        return list(map(self.find, self.parent_))


def distance_1(title1, title2, num_of_words):
    # simple vec distance
    id1 = 0
    id2 = 0
    vec = []
    while id1 < len(title1) and id2 < len(title2):
        x = y = 0
        word1 = title1[id1][0]
        word2 = title2[id2][0]
        if word1 <= word2:
            x = title1[id1][1]
            id1 += 1
        if word1 >= word2:
            y = title2[id2][1]
            id2 += 1
        vec.append(abs(x - y))
    while id1 < len(title1):
        vec.append(title1[id1][1])
        id1 += 1
    while id2 < len(title2):
        vec.append(title2[id2][1])
        id2 += 1
    return sqrt(sum(map(lambda x: x * x, vec)))


def distance_2(title1, title2, num_of_words):
    # manhattan vec distance
    id1 = 0
    id2 = 0
    vec = []
    while id1 < len(title1) and id2 < len(title2):
        x = y = 0
        word1 = title1[id1][0]
        word2 = title2[id2][0]
        if word1 <= word2:
            x = title1[id1][1]
            id1 += 1
        if word1 >= word2:
            y = title2[id2][1]
            id2 += 1
        vec.append(abs(x - y))
    while id1 < len(title1):
        vec.append(title1[id1][1])
        id1 += 1
    while id2 < len(title2):
        vec.append(title2[id2][1])
        id2 += 1
    return sum(vec)


def distance_3(title1, title2, num_of_words):
    # Pearson
    x = [0] * num_of_words
    for elem in title1:
        x[elem[0] - 1] += elem[1]
    y = [0] * num_of_words
    for elem in title2:
        y[elem[0] - 1] += elem[1]
    x_m = sum(x) / len(x)
    y_m = sum(y) / len(y)
    x = [x[_] - x_m for _ in range(len(x))]
    y = [y[_] - y_m for _ in range(len(x))]
    a = sum([x[_] * y[_] for _ in range(len(x))])
    b = sum([x[_] * x[_] for _ in range(len(x))])
    c = sum([y[_] * y[_] for _ in range(len(x))])
    if a == 0:
        return 1
    return 1 - (a / sqrt(b * c))


def cluster(ver, eds, num_of_clusters):
    # return DisjointSetUnion of edges
    res = DisjointSetUnion(len(ver))
    # Your code here
    cnt = 0
    sorted_edges = sorted(eds, key=lambda x: x[0])
    for e in sorted_edges:
        if res.find(e[1]) != res.find(e[2]):
            cnt += 1
            res.union_sets(e[1], e[2])
            if cnt == len(ver) - num_of_clusters:
                break
    return res


def cluster1(ver, eds, num_of_clusters):
    res = [-1] * len(ver)
    clusters = set()
    a = random.randint(0, len(ver) - 1)
    #a = 0
    clusters.add(a)
    res[a] = a
    while len(clusters) != num_of_clusters:
        best_cnt = -1
        new = 0
        for i in range(len(ver)):
            if i not in clusters:
                cnt = 0
                for j in clusters:
                    cnt += eds[i][j]
                if cnt > best_cnt:
                    best_cnt = cnt
                    new = i
        clusters.add(new)
        res[new] = new
    for i in range(len(ver)):
        if i not in clusters:
            best_d = -1
            cl = -1
            for j in clusters:
                new_d = eds[i][j]
                if best_d == -1 or best_d > new_d:
                    cl = j
                    best_d = new_d
            res[i] = cl
    return res


t_start = time.time()

# input file
filename = 'test_400.txt'

# function of distance
distance = distance_3

# clusters num
clusters_num = 5

# word freq param
min_count = 2
max_count = 10

vertices = []
with open(filename) as f:
    doc_num = int(f.readline())
    words_num = int(f.readline())
    lines_num = int(f.readline())
    words_count = [0] * words_num
    vertices = [[] for _ in range(doc_num)]
    for i in range(lines_num):
        docID, wordID, count = map(int, f.readline().split())
        words_count[wordID - 1] += count
        if count <= max_count:
            vertices[docID - 1].append([wordID, count])

new_v = []
for elem in vertices:
    new_v.append([])
    for k in elem:
        if min_count <= words_count[k[0] - 1] <= max_count:
            new_v[-1].append(k)
    if len(new_v[-1]) == 0:
        print("Warning: vertex without words")
vertices = new_v
print('Vertices:', len(vertices))

edges = [(distance(vertices[i], vertices[j], words_num), i, j) for i in range(len(vertices) - 1) for j in range(i + 1, len(vertices))]
print('Edges:', len(edges))

edges1 = [[0 for i in range(len(vertices))] for j in range(len(vertices))]
for i in range(len(vertices) - 1):
    for j in range(i + 1, len(vertices)):
        d = distance(vertices[i], vertices[j], words_num)
        edges1[i][j] = d
        edges1[j][i] = d

#components = cluster(vertices, edges, clusters_num)
components1 = cluster1(vertices, edges1, clusters_num)

comp = components1
#comp = components.result()
d = dict()
for i in range(len(comp)):
    if comp[i] not in d:
        d[comp[i]] = [i + 1]
    else:
        d[comp[i]].append(i + 1)
for elem in d:
    print("Cluster", elem, ":", len(d[elem]), "elems", ":", d[elem])

print("Time:", time.time() - t_start)
