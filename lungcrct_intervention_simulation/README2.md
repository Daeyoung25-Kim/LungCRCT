## LungCRCT_Causal_Intervention_Simulation  


<br> <div align="justify"> For practical implementations of the LungCRCT model in simulating lung cancer treatment progressions or cancer progression itself, a simplified causal intervention pipeline and its methodology were introduced in this paper. Overall procedures of the intervention pipeline can be summarized as Figure 1.
</div>


<br> <div align="center">
  <img width="760" height="727" alt="image" src="https://github.com/user-attachments/assets/d6564100-bfb1-46dd-9c01-78d3882d9584" />
</div>
(*Mild updates in hyperparamters for LungCRCT training due to tensorflow package upgrade)

<br> <div align="justify"> Based on analysis in [1], the LungCRCT framework was found to not only return valid latent causal representations( Z<sub>cs</sub> ) of tumor, angiogenesis and lymph node enlargements, but also extract the underlying causal relationships via a weighted causal adjacency matrix, leading to the conclusion that causal effects resulting from alteration in individual ground truth lung cancer factors can be approximated through intervening features within the Z<sub>cs</sub> space from LungCRCT under the assumption of J, Pearl’s Structural Causal Model(SCM)[2] and the Structural Causal Layer (intervention) approach from CausalVAE [3]. Thus, for lung cancer intervention simulation, fitted LungCRCT’s Graph Autoencoder structures and the Convolutional VAE Decoder part were utilized to create the basis of the overall pipeline. Furthermore, to enhance significance of causal inference from fitted network structures, residual variable’s μ vector was approximated and added to the results of intervention (ẑ). As Graph Autoencoder-based causal learning approaches do not consider noise distributions in causality extractions, using residual values: (z-ẑ) to estimate bias terms and feature-wise noise distributions can be crucial for SCM assumptions to be sufficiently satisfied. (Post-noise centering for SCM assumption satisfaction). Within LungCRCT based intervention simulations, bias terms: μ are approximated using the sample mean from residuals: r. Adding the bias term: μ to ẑ, significance of intervention within causal latent space was enhanced, leading to better visual simulation results for practical implementations. </div>


$$
r= z-\hat{z} ; \quad r: \text{residual}, \quad r= \mu+\epsilon ; \quad E(\epsilon)=0, \epsilon: \text{noise variable}
$$

<br> <div align="justify"> Meanwhile, in order to overcome the limitations deduced from implementing lightweight ConvTranspose layers or using VAE for image reconstruction tasks, an additional Upscaling network was constructed and later applied to output images from the LungCRCT decoder (Figure 2). (For simplification, Upscaling structures were also constructed ‘lightweight’).
</div>

<br><div align="center"><img width="756" height="380" alt="image" src="https://github.com/user-attachments/assets/510579bd-e87e-4ec0-9099-2562b0b6dda3" />
</div>

<br> <div align="justify"> First applying a shallow U-Net based denoising convolutional network, and then applying a Python cv2 package based CLAHE(Contrast Limited Adaptive Histogram Equalization) with parameters set as clipLimit=1.8 and tileGridSize=(7,7), original reconstructed images from the LungCRCT decoder were upscaled to enhance contrast and deblur noise. Resulting simulation procedures for intervention visualization can be summarized as TABLE I.
</div>

<br><div align="center"><img width="577" height="457" alt="image" src="https://github.com/user-attachments/assets/dd12fe09-172c-465f-8bc0-6082e510c8be" /> 
</div>

<br><br><div align="justify"> Within the proposed framework, visual results from intervention simulation are returned as MP4 files to help individuals comprehend the consequential causal alteration of lung cancer features in a continuous aspect. For example, the first MP4 exhibits an intervention analysis case where tumor elimination is assumed for cancer treatment (Z<sub>cs,0</sub>: -0.4901 (original value for tumor variable representation) to -0.1). As the intervention process progresses, tumor size is visibly being reduced due to varying Z<sub>cs,0</sub>, along with clear reductions regarding size of enlarged lymph nodes and angiogenesis within the cancerous lung (emphasized by red circles) due to causal intervention effects. Meanwhile, the second MP4 exhibits an intervention analysis case where tumor growth is assumed from a non-cancerous healthy lung state. (Tumor variable Z<sub>cs,0</sub> is intervened from 0.0279 (original value) to -0.3). As intervention progresses, consistent tumor growth within the left lung is observed, with enlargements in the lymph node clearly being developed (emphasized by red circles). 
</div>


<br><table width="100%">
  <tr>
    <td width="50%" align="center">
      <video src="https://github.com/user-attachments/assets/765b7f4e-7c52-46c9-8c83-ad0f13e8e5bd" width="300px" autoplay loop muted playsinline controls></video>
    </td>
    <td width="50%" align="center">
      <video src="https://github.com/user-attachments/assets/50e8b9fc-9740-4fb8-b9f0-3597421ade5c" width="300px" autoplay loop muted playsinline controls></video>
    </td>
  </tr>
</table>

<br> <div align="center"><img width="1235" height="617" alt="image" src="https://github.com/user-attachments/assets/cd825410-bbb6-464a-b616-65b246ecb2d1" />
</div>



<br><br> <div align="justify"> As exhibited in MP4 examples, not only latent causal factor extractions for accurate, lightweight downstream causal lung cancer detections, but also valid visual treatment simulations based on intervention analysis can be achieved with high significance under the proposed LungCRCT-based causal intervention simulation framework. Free from spurious correlations by incorporating latent causal representation learning mechanisms to vision model architectures, LungCRCT expands its field from reliable causal vision detection tasks to visualizing lung tumor treatment trajectories or lung cancer progression.
</div>

<br><br> **Remaining Limitations**: Due to the reliance on extremely lightweight CNN structures (from LungCRCT frameworks to Upscaling Network architectures), lung tumor intervention simulation results were found not to be free from the visual limitations found in previous VAE-based works (ex: image blurring). To guarantee success in real world medical field applications, additional decoder pipelines that assume advanced image reconstruction network structures, such as diffusion models or GAN, are recommended.

For more information regarding LungCRCT please check the original paper: https://arxiv.org/abs/2601.18118

If you find LungCRCT helpful, please cite the original article(https://arxiv.org/pdf/2601.18118)
- D, Kim, “LungCRCT: Causal Representation based Lung CT Processing for Lung Cancer Treatment”, arXiv:2601.18118, pp. 1-11, 2026.

<br> <div align="justify"> **As a frontier model in Causal Medical Vision AI analysis, LungCRCT has been aiming to go beyond spurious correlations in lung cancer analysis, potentially leading to the possibility of Causal AI-based personal treatment simulation through medical images.** 
</div>

<br> Thank you!


#### Reference
[1] D, Kim, “LungCRCT: Causal Representation based Lung CT Processing for Lung Cancer Treatment”, arXiv:2601.18118, pp. 1-11, 2026.

[2] J, Pearl, Causality, Cambridge university press, 2009.

[3] M, Yang et al., "CausalVAE: Disentangled Representation Learning via Neural Structural Causal Models", CVPR 2021, pp. 9593-9602, 2021.
