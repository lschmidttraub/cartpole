import mujoco
import mujoco.viewer
import os

os.environ["LIBGL_ALWAYS_SOFTWARE"]="1"

with open('envs/cartpole.xml', 'r') as f:
    xml = f.read()

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)  # Reset to "start" keyframe

with mujoco.Renderer(model) as renderer:
    renderer.update_scene(data)

    mujoco.viewer.launch(model, data)
