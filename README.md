## ML-interpolation-Master-Thesis
# Variational Autoencoders for Polyphonic Music Interpolation
This is my Master Thesis, submitted to National Tsing Hua University (Taiwan, July 2020).

### **Abstract**

This thesis aims to use Machine Learning techniques to solve the novel problem of music interpolation composition. Two models based on Variational Autoencoders (VAEs) are proposed to generate a suitable polyphonic harmonic bridge between two given songs, smoothly changing the pitches and dynamics of the interpolation. The interpolations generated by the first model surpass a Random baseline data and a bidirectional LSTM approaches and its performance is comparable to the current state-of-the-art. The novel architecture of the second model outperforms the state-of-the-art interpolation approaches in terms of reconstruction loss by using an additional neural network for direct estimation of the interpolation encoded vector. Furthermore, the _Hsinchu Interpolation MIDI Dataset_ was created, making both models proposed in this thesis more efficient than previous approaches in the literature in terms of computational and time requirements during training. Finally, a quantitative user study was done in order to ensure the validity of the results.

### **What do we mean by music interpolation?**

“Interpolation is a type of **estimation**, a method of constructing **new data** points within the range of a discrete set of known data points” (Fleetwood, 1991)

In traditional Machine Learning, the generation of music is conditioned on the past events. But what if we could **condition the music generation on both past and future events**? We would input a begin track and an end track of 10 seconds each to our model, obtaining the middle (or interpolation) track of also 10 seconds as output, whose pitches and dynamics match both given tracks.

<p align="center">
  <img width="550" src="https://github.com/pablomp3/ML-interpolation-Master-Thesis/blob/master/images/interpolation_definition.jpg">
</p>

### **What is polyphonic music and how to model it?**

In **monophonic** music, every timestep or time unit contains one single note. On the other hand, in **polyphonic** music, every timestep contains several notes, forming chords that make the composition richer. We use MIDI (Musical Instrument Digital Interface) to represent the music in a symbolic way, instead of using the raw waveform (which is computationally expensive to manipulate). Each timestep of a song is represented as a **vector of 64 binary elements**, where each binary element represents one piano key (or one note or pitch), 1 meaning _note on_ and 0 meaning _note off_.

<p align="center">
  <img width="550" src="https://github.com/pablomp3/ML-interpolation-Master-Thesis/blob/master/images/mono_vs_polyphonic.jpg">
</p>

<p align="center">
  <img width="550" src="https://github.com/pablomp3/ML-interpolation-Master-Thesis/blob/master/images/polyphonic_modelling.jpg">
</p>

### Dataset

A new MIDI dataset based on the Lahk MIDI Dataset has been created: the _Hsinchu Interpolation MIDI Dataset_. This dataset contains only very valuable interpolation segments, where the begin track and the end track are very different (simulating a style transfer within the same human composition). The begin and end tracks similarity has been evaluated with a neural network (binary classification problem). The _Hsinchu Interpolation MIDI Dataset_ contains 30,830 segments of 30 seconds each.

### Experiments

Four experiments have been done in this thesis. Experiments 1 and 2 are the **baselines**. Experiments 03 and 04 are **proposed novel models** based on Variational Autoencoders to solve the interpolation problem:

01. Random Data
02. Bi-LSTM
03. VAE (Variational Autoencoder)
04. VAE+NN (Variational Autoencoder + Neural Network)

#### Experiment 3. VAE: interpolation done with linear sampling of latent space. Steps:

01. Encode begin track and end track with VAE to obtain z_begin and z_end, respectively:
<p align="center">
  <img width="460" src="https://github.com/pablomp3/ML-interpolation-Master-Thesis/blob/master/images/begin_end_encoding.jpg">
</p>

02. Average vectors z_begin and z_end to obtain the interpolation encoded vector z_interpolation:
<p align="center">
  <img width="460" src="https://github.com/pablomp3/ML-interpolation-Master-Thesis/blob/master/images/z_interpolation_by_average.jpg">
</p>

03. Decode the interpolation encoded vector z_interpolation to obtain the interpolation track:
<p align="center">
  <img width="460" src="https://github.com/pablomp3/ML-interpolation-Master-Thesis/blob/master/images/z_interpolation_decoding.jpg">
</p>

04. Ideally, the reconstructed interpolation track has to be identical to the original interpolation track (ground truth):
<p align="center">
  <img width="460" src="https://github.com/pablomp3/ML-interpolation-Master-Thesis/blob/master/images/original_vs_reconstructed_input.jpg">
</p>

#### Experiment 4. VAE+NN: interpolation done with direct estimation of the interpolation encoded vector. Steps:

01. Encode begin track and end track with VAE to obtain z_begin and z_end, respectively:
<p align="center">
  <img width="460" src="https://github.com/pablomp3/ML-interpolation-Master-Thesis/blob/master/images/begin_end_encoding.jpg">
</p>

02. Use the novel neural network (NN) approach to directly estimate the interpolation encoded vector z_interpolation based on z_begin and z_end:
<p align="center">
  <img width="460" src="https://github.com/pablomp3/ML-interpolation-Master-Thesis/blob/master/images/z_interpolation_by_nn.jpg">
</p>

03. Decode the interpolation encoded vector z_interpolation to obtain the interpolation track:
<p align="center">
  <img width="460" src="https://github.com/pablomp3/ML-interpolation-Master-Thesis/blob/master/images/z_interpolation_decoding.jpg">
</p>

04. Ideally, the reconstructed interpolation track has to be identical to the original interpolation track (ground truth):
<p align="center">
  <img width="460" src="https://github.com/pablomp3/ML-interpolation-Master-Thesis/blob/master/images/original_vs_reconstructed_input.jpg">
</p>

Full architecture of the novel VAE+NN model proposed in this thesis:

<p align="center">
  <img width="550" src="https://github.com/pablomp3/ML-interpolation-Master-Thesis/blob/master/images/VAE%2BNN_architecture.jpg">
</p>

### Results

