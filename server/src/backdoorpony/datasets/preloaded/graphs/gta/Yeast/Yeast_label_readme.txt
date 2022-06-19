README for data set Yeast

=== Usage ===

This folder contains the following comma separated text files 
(replace DS by the name of the dataset):

n = total number of nodes
m = total number of edges
N = number of graphs

(1) 	DS_A.txt (m lines) 
	sparse (block diagonal) adjacency matrix for all graphs,
	each line corresponds to (row, col) resp. (node_id, node_id)

(2) 	DS_graph_indicator.txt (n lines)
	column vector of graph identifiers for all nodes of all graphs,
	the value in the i-th line is the graph_id of the node with node_id i

(3) 	DS_graph_labels.txt (N lines) 
	class labels for all graphs in the dataset,
	the value in the i-th line is the class label of the graph with graph_id i

(4) 	DS_node_labels.txt (n lines)
	column vector of node labels,
	the value in the i-th line corresponds to the node with node_id i

There are OPTIONAL files if the respective information is available:

(5) 	DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)
	labels for the edges in DS_A_sparse.txt 

(6) 	DS_edge_attributes.txt (m lines; same size as DS_A.txt)
	attributes for the edges in DS_A.txt 

(7) 	DS_node_attributes.txt (n lines) 
	matrix of node attributes,
	the comma seperated values in the i-th line is the attribute vector of the node with node_id i

(8) 	DS_graph_attributes.txt (N lines) 
	regression values for all graphs in the dataset,
	the value in the i-th line is the attribute of the graph with graph_id i


=== Node Label Conversion === 
Node labels were converted to integer values using this map:

Component 0:
Cu 0
O 1
N 2
C 3
Y 4
Nd 5
Pt 6
P 7
Sn 8
Fe 9
S 10
Cl 11
Eu 12
F 13
Ti 14
Zr 15
Hf 16
Br 17
Na 18
Hg 19
La 20
Ce 21
Zn 22
Mn 23
Co 24
Ni 25
I 26
Au 27
Pb 28
Pd 29
Ge 30
K 31
Tl 32
As 33
Ru 34
Cd 35
Ga 36
Se 37
Bi 38
Sb 39
Si 40
B 41
Rh 42
Mo 43
Nb 44
In 45
Os 46
Ag 47
Gd 48
Ba 49
Er 50
W 51
V 52
Dy 53
Sm 54
Te 55
Cr 56
Mg 57
Ir 58
Li 59
Po 60
Al 61
Re 62
Fr 63
Ta 64
Cs 65
Ho 66
Pr 67
Tb 68
Ac 69
Be 70
Ca 71
Rb 72
Am 73



Edge labels were converted to integer values using this map:

Component 0:
	0	1
	1	2
	2	3

=== References ===
Source: https://sites.cs.ucsb.edu/~xyan/dataset.htm

PubChem website (http://pubchem.ncbi.nlm.nih.gov).  PubChem provides information on the biological activities of small molecules, containing the bioassay records for anti-cancer screen tests with different cancer cell lines. Each dataset belongs to a certain type of cancer screen with the outcome active or inactive. From these screen tests, we collected 11 graph datasets with active and inactive labels.

Name 	Assay  ID 	Size 	Tumor Description
MCF-7 	83 	27770 	Breast
MOLT-4 	123 	39765 	Leukemia
NCI-H23 	1 	40353 	Non-Small Cell Lung
OVCAR-8 	109 	40516 	 Ovarian
P388 	330 	41472 	Leukemia
PC-3 	41 	27509 	Prostate
SF-295 	47 	40271 	Central Nerv Sys
SW-620 	81 	40532 	Colon
SN12C 	145 	40004 	Renal
UACC-257 	33 	39988 	Melanoma
Yeast 167 	167 	79601 	Yeast anticancer


