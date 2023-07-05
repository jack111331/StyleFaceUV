Code:
	stylegan2_to_3d.py 								-- 訓練3D Generator
	stylecode_to_3Dcoeff.py 						-- 訓練Stylecode to 3D Coeffs
	latent_to_Pose_Scalar.py 						-- 訓練Stylecode to Pose Scalar , 控制StyleGAN2生成對稱角度照片
	latent_to_Multi_Pose_Scalar.py  				-- 訓練Stylecode to Pose Scalar , 控制StyleGAN2生成各種角度照片
	inference_styleuv-shape-tex.py  				-- generate images from pre-trained 3D generator
	inference_styleuv-shape-tex-gif.py  			-- generate gifs from pre-trained 3D generator
	MODEL.py 										-- 存放各式model，如stylecode_to_3Dcoeff , latent_to_pose_scalar, Pure_3DMM
	MODEL_UV.py 									-- 存放能夠讀取UV Map的3DMM : Pure_3DMM_UV
	model.py 										-- 存放2D StyleGAN2 model , 有小修改來直接input stylecode
	model_uv.py 									-- 存放3D Generator的model , 修改成output UV tex & shape maps
	losses.py 										-- 存放各式loss
	DATA.py 										-- 存放各式Dataset
	train_boundary.py 								-- 訓練Pose direction

Directory:
	BFM/ 											-- 放一些定義好的3DMM Model (Basel Face Model 2009)
	checkpoints/ 									-- checkpoints of stylecode_to_3Dcoeff.py
	checkpoints_final/ 								-- checkpoints of stylegan2_to_3d.py
	checkpoints_latent2posescalar-new-sym/ 			-- checkpoints of latent_to_Pose_Scalar.py
	checkpoints_latent2posescalar-new_multi-pose/ 	-- checkpoints of latent_to_Multi_Pose_Scalar.py
	models/ 										-- checkpoints of StyleGAN2
	pkls/
		3DMMparam-new.pkl 							-- fitting results of StyleGAN2-generated images   size : (39592, 257)
		pose_direction-new.pkl 						-- pose direction found by train_boundary.py 	   size : (1, 7168)
		tensor-new.pkl 								-- fixed stylecodes                                size : (39592, 14, 512)
	train_stylegan_images/ 							-- StyleGAN2-generated images + mask.png           size : (39592+1)

Related Github website:
https://github.com/rosinality/stylegan2-pytorch
https://github.com/ascust/3DMM-Fitting-Pytorch
https://github.com/genforce/interfacegan
https://github.com/anilbas/3DMMasSTN

其他沒提到的應該是StyleGAN2本身需要用到的程式，或者是目前用不到的東西，但還是留著以防萬一。