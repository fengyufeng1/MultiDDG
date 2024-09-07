# Download four files from Google Drive 
https://drive.google.com/file/d/1ku0BNuzc5Hq5IfDCxRLNys3om8nbFF4F/view?usp=sharing
(eg.Atom_edge_index, Atom_feature, Edge_index, Feature_SeqVec)

# Unzip the folder
```bash
tar -xf Feature_S1131-S4169-S8338-S645.tar
cd Feature_S1131-S4169-S8338-S645
mkdir -p Atom_edge_index && cat Atom_edge_index.tar.gz.* | tar -xzv -C Atom_edge_index
mkdir -p Atom_feature && cat Atom_feature.tar.gz.* | tar -xzv -C Atom_feature
mkdir -p Edge_index && cat Edge_index.tar.gz.* | tar -xzv -C Edge_index
mkdir -p Feature_SeqVec && cat Feature_SeqVec.tar.gz.* | tar -xzv -C Feature_SeqVec
```


