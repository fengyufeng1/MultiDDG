# Download files from Google Drive
Feature_2M.tar: https://drive.google.com/file/d/1iQpseiGjO8hg0fLVRVWRgMOkGsvj6Osf/view?usp=sharing
Feature_CAPRI.tar: https://drive.google.com/file/d/13PpelF4C1z5F6XmFe95MlrVdv7jY-YL2/view?usp=sharing
Feature_M1340-M595.tar: https://drive.google.com/file/d/1DV5sv92YJQSs9fm4u3DTbAsPDzFCdKob/view?usp=sharing
Feature_M1707.tar: https://drive.google.com/file/d/1j9e5kZf2q0emDaiNVFb3aU8YM02cojf_/view?usp=sharing
Feature_S33.tar: https://drive.google.com/file/d/1mE3_ochYO-bTwcdSIvC_SQ47RBrIsFCp/view?usp=sharing
Feature_S285.tar: https://drive.google.com/file/d/12NdvqL5nK69HSA2q0yGW4nVsERo57uT6/view?usp=sharing
Feature_S1131-S4169-S8338-S645.tar: https://drive.google.com/file/d/1ku0BNuzc5Hq5IfDCxRLNys3om8nbFF4F/view?usp=sharing
Feature_S4191.tar: https://drive.google.com/file/d/1X7nCId1ZXv2aM4oROL5RsBDDci1DPci3/view?usp=sharing

# Unzip the folder
```bash
tar -xf ./Feature_2M.tar
cd ./Feature_2M
cat ./Atom_edge_index.tar.gz.* | tar -xzv -C ./
cat ./Atom_feature.tar.gz.* | tar -xzv -C ./
cat ./Edge_index.tar.gz.* | tar -xzv -C ./
cat ./Feature_SeqVec.tar.gz.* | tar -xzv -C ./
```
