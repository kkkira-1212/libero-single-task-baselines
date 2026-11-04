"""Map raw LIBERO observation to official BAKU act input, matching flipped PKL images."""
import numpy as np
def online_observation(obs):
    result={}
    for key,source in (("pixels","agentview_image"),("pixels_egocentric","robot0_eye_in_hand_image")):
        image=np.asarray(obs[source])
        if image.dtype!=np.uint8 or image.shape!=(128,128,3):
            raise ValueError("Expected raw128x128uint8RGB: "+source)
        result[key]=np.ascontiguousarray(image[::-1].transpose(2,0,1))
    result["proprioceptive"]=np.concatenate([obs["robot0_joint_pos"],obs["robot0_gripper_qpos"]]).astype(np.float32)
    if result["proprioceptive"].shape!=(9,) or not np.isfinite(result["proprioceptive"]).all():
        raise ValueError("Invalid9Dstate")
    return result
def identity_stats():
    # Mirrors official LIBERO BCDataset stats, not DP limits normalization.
    return {"actions":{"min":0,"max":1},"proprioceptive":{"min":0,"max":1}}
