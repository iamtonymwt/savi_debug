SAVi (JAX) Run Scripts:

python -O -m savi.main --config savi/configs/movi/savi_conditional_medium.py --workdir tmp/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hci-p920-3/anaconda3/envs/SAVi/lib

pip install "jax[cuda11_cudnn82]" 


######

SAVi-pytorch Run Scripts:

源代码：
改batchsize，dir
注释掉了visualize
python -m savi.main 

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 10000)])


修改代码：
* datasets加入了dataloader
* 改了dataloader的dir,getitem顺序,返回值类型
* datasets移入了数据集
* 删掉了最后一张图片
* dataparallel:
	* import
	* args(tfds注释,maxinstance,eval_slice_size)
	* dataloader
	* builddatasets
	* resetitr注释掉
* encoder embedding size修改，可以使得输入尺寸没问题，可以正常算loss
* steps = 33750
* 改了decoder的convTranspose2d
* misc mask = none
* visualize isample = 2
* train的evaluate参数加了个evaluator
* 改了一下gloal_step的计算
* 输出删f
* 数据读入加速了
* image减除，对应visialize
* 改了resume的一套
* wandb:baa5547df219d350a0ce90782b5e68a622f42a05
* scheduler初始化提到了run，传入epoch
* 改了loss，和metrics
* 在decoder的dict里加一个segmentation
* 将metric换成了miou
* 新加入了fold和seq的args信息
* 加入了.sh文件来跑四次
* 加入了visual mode 需要手工设置batch_size、seq、resume_from(.pt文件的路径)
	* python -m savi.main --visual --resume_from "experiments14/13/snapshots/79.pt" --batch_size 1 --seq 1
	* python -m savi.main --wandb --group size_512 --exp 15
* 加了.gitignore傳上了git
* 修改了dataloader的seq讀取順序
* 改分辨率512 640
	* batch, seq, steps
	* dataloader
	* factory:encoderPE,decoder(Backbone,broadcast,PE)
	* misc的load_snapshot(傳入了args，解決了不匹配問題)
* semi(測試過label計算無問題)
	* dataloader:路徑(不應該改image)、label、圖片數量
	* trainer:加label、label對接
	* misc:新的semiloss，加入label對接,改了label的使用方式
	* factory的loss
	* 加入的算loss的類libmtl.py沒用，暫時是loss = recon 0.3 + focal 0.7
* nbb
	* network文件夾
	* 新的snapshots（修改snapshots路徑、不load mobilenet）
	* factory import, 修改encoder
	* batch改成了2，memory問題
	* encoder出來是64 80，attention可視化時候差值成了512 640
	* 重新計算steps和epochs
	* 拆分loss傳上wandb
	* 改decoder的resolution
	* 改了dataloader，支持了18個seq
	* ok dataset進行了更新
	* ok 由於fold不同load進來的base不同，更新了train.sh
	* 修改了initializer，直接把slot高斯分布
	* unlabeled的data計算flow loss時用predict seg來mask gt flow
	* OK 小改了一下decoder的輸出 
	* 修改了snap_load，同時load encoder和attention
	* utils裏的batch_broadcast有問題，改了一下，加上了ParamStateInit的選項
	* 改了corrector和predictor的channels，滿足SAVi-L的配置
	* attention裏一個layer_norm和gru size的小bug
	* encoder backbone出來的size變爲了256，相應的PE和MLP也都改爲了256。目前又改成了304
	














