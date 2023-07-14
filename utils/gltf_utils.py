import numpy as np
import pygltflib
import os
from PIL import Image
import torch

def output_mesh_to_gltf(vertices, indices, uv_coord, diffuse_map, save_path):
    vertices = vertices.squeeze(0).detach().cpu().numpy().astype('float32')
    indices = indices.squeeze(0).detach().cpu().numpy().astype('uint16')
    uv_coord = uv_coord.squeeze(0).squeeze(0).detach().cpu().numpy().astype('float32')
    diffuse_map = diffuse_map.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    diffuse_map_img = (diffuse_map * 255).astype(np.uint8)
    path = os.path.join('gradio_tmp', "test.png")
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    Image.fromarray(diffuse_map_img).save(path)

    # Scale so that the vertices can be visible in gltf viewer
    vertices *= 35.0
    triangles_binary_blob = indices.flatten().tobytes()
    points_binary_blob = vertices.flatten().tobytes()
    uv_coord_binary_blob = uv_coord.flatten().tobytes()

    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[
            pygltflib.Mesh(
                primitives=[
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(POSITION=0, TEXCOORD_0=1), 
                        indices=2,
                        material=0
                    )
                ]
            )
        ],
        accessors=[
            pygltflib.Accessor(
                bufferView=0,
                componentType=pygltflib.FLOAT,
                count=len(vertices),
                type=pygltflib.VEC3,
                max=vertices.max(axis=0).tolist(),
                min=vertices.min(axis=0).tolist(),
            ),
            pygltflib.Accessor(
                bufferView=1,
                componentType=pygltflib.FLOAT,
                count=len(uv_coord),
                type=pygltflib.VEC2,
                max=uv_coord.max(axis=0).tolist(),
                min=uv_coord.min(axis=0).tolist(),
            ),
            pygltflib.Accessor(
                bufferView=2,
                componentType=pygltflib.UNSIGNED_SHORT,
                count=indices.size,
                type=pygltflib.SCALAR,
                max=[int(indices.max())],
                min=[int(indices.min())],
            ),
        ],
        bufferViews=[
            # Mesh vertices, indices, uv coordinate
            pygltflib.BufferView(
                buffer=0,
                byteLength=len(points_binary_blob),
                target=pygltflib.ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(points_binary_blob),
                byteLength=len(uv_coord_binary_blob),
                target=pygltflib.ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(points_binary_blob) + len(uv_coord_binary_blob),
                byteLength=len(triangles_binary_blob),
                target=pygltflib.ELEMENT_ARRAY_BUFFER,
            ),
        ],
        buffers=[
            pygltflib.Buffer(
                byteLength=len(points_binary_blob) + len(uv_coord_binary_blob) + len(triangles_binary_blob)
            )
        ],
        materials=[
            pygltflib.Material(
                pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                    baseColorTexture=pygltflib.TextureInfo(index=0),
                ), 
                alphaCutoff=None,
            )
        ],
        textures=[
            pygltflib.Texture(
                source=0
            )
        ],
    )
    image = pygltflib.Image()
    image.uri = path
    gltf.images.append(image)
    gltf.convert_images(pygltflib.ImageFormat.DATAURI, override=True)

    gltf.set_binary_blob(points_binary_blob + uv_coord_binary_blob + triangles_binary_blob)
    gltf.convert_buffers(pygltflib.BufferFormat.DATAURI, override=True)
    gltf.save(save_path)
