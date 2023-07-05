import torch
import torch.nn as nn
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    blending,
    Textures
)
import torch.nn.functional as F

class Pure_3DMM(nn.Module):
    def __init__(self, face_model, focal=1015, image_size=256, device='cuda'):
        super(Pure_3DMM, self).__init__()
        self.facemodel = face_model
        self.focal = focal
        self.img_size = image_size
        self.device = torch.device(device)
        self.renderer = self.get_renderer(self.device)
        self.kp_inds = torch.tensor(self.facemodel['keypoints'] - 1).squeeze().long()
        meanshape = nn.Parameter(torch.from_numpy(self.facemodel['meanshape']).float(), requires_grad=False)
        self.register_parameter('meanshape', meanshape)
        idBase = nn.Parameter(torch.from_numpy(self.facemodel['idBase']).float(), requires_grad=False)
        self.register_parameter('idBase', idBase)

        exBase = nn.Parameter(torch.from_numpy(self.facemodel['exBase']).float(), requires_grad=False)
        self.register_parameter('exBase', exBase)

        meantex = nn.Parameter(torch.from_numpy(self.facemodel['meantex']).float(), requires_grad=False)
        self.register_parameter('meantex', meantex)

        texBase = nn.Parameter(torch.from_numpy(self.facemodel['texBase']).float(), requires_grad=False)
        self.register_parameter('texBase', texBase)

        tri = nn.Parameter(torch.from_numpy(self.facemodel['tri']).float(), requires_grad=False)
        self.register_parameter('tri', tri)

        point_buf = nn.Parameter(torch.from_numpy(self.facemodel['point_buf']).float(), requires_grad=False)
        self.register_parameter('point_buf', point_buf)

    def get_renderer(self, device):
        R, T = look_at_view_transform(10, 0, 0)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01, zfar=50,
                                        fov=2 * np.arctan(self.img_size // 2 / self.focal) * 180. / np.pi)

        lights = PointLights(device=device, location=[[0.0, 0.0, 1e5]], ambient_color=[[1, 1, 1]],
                             specular_color=[[0., 0., 0.]], diffuse_color=[[0., 0., 0.]])

        raster_settings = RasterizationSettings(
            image_size=self.img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        blend_params = blending.BlendParams(background_color=[0, 0, 0])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            # shader = SoftSilhouetteShader(
            #     device=device,
            #     cameras=cameras,
            #     lights=lights,
            #     blend_params=blend_params
            # )
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights,
                blend_params=blend_params
            )
        )
        return renderer

    @staticmethod
    def Split_coeff(coeff):
        id_coeff = coeff[:, :80]  # identity(shape) coeff of dim 80
        ex_coeff = coeff[:, 80:144]  # expression coeff of dim 64
        tex_coeff = coeff[:, 144:224]  # texture(albedo) coeff of dim 80
        angles = coeff[:, 224:227]  # ruler angles(x,y,z) for rotation of dim 3
        gamma = coeff[:, 227:254]  # lighting coeff for 3 channel SH function of dim 27
        translation = coeff[:, 254:]  # translation coeff of dim 3

        return id_coeff, ex_coeff, tex_coeff, angles, gamma, translation

    # compute face shape with identity and expression coeff, based on BFM model
    # input: id_coeff with shape [1,80]
    #        ex_coeff with shape [1,64]
    # output: face_shape with shape [1,N,3], N is number of vertices
    def Shape_formation(self, id_coeff, ex_coeff):
        n_b = id_coeff.size(0)

        face_shape = torch.einsum('ij,aj->ai', self.idBase, id_coeff) + \
                     torch.einsum('ij,aj->ai', self.exBase, ex_coeff) + self.meanshape
        # about einsum , we can see it as (ij,aj)->ai
        # idBase : (107127, 80)
        # exBase : (107127, 64)
        face_shape = face_shape.view(n_b, -1, 3)
        face_shape = face_shape - self.meanshape.view(1, -1, 3).mean(dim=1, keepdim=True)
        # 最後扣掉所有vertex的mean , shape:(1,1,3)
        return face_shape

    # compute vertex texture(albedo) with tex_coeff
    # input: tex_coeff with shape [1,N,3] (X)
    # input: tex_coeff with shape [1,80]
    # output: face_texture with shape [1,N,3], RGB order, range from 0-255
    def Texture_formation(self, tex_coeff):
        n_b = tex_coeff.size(0)
        # print(self.texBase.size(), self.meantex.size(), tex_coeff.size())
        face_texture = torch.einsum('ij,aj->ai', self.texBase, tex_coeff) + self.meantex

        face_texture = face_texture.view(n_b, -1, 3)
        return face_texture

    # compute vertex normal using one-ring neighborhood
    # input: face_shape with shape [1,N,3]
    # output: v_norm with shape [1,N,3]
    # face shape僅是每一個vertex在3D空間中的位置
    # face_id / self.tri 才是哪三個vertex定義一個三角形網格，而這是預先定義在BFM model中的!
    def Compute_norm(self, face_shape):
        face_id = self.tri.long() - 1  # vertex index for each triangle face, with shape [F,3], F is number of faces
        point_id = self.point_buf.long() - 1  # adjacent face index for each vertex, with shape [N,8], N is number of vertex
        # print(face_id.size(), point_id.size()) #(70789,3) , (35709, 8)
        shape = face_shape
        v1 = shape[:, face_id[:, 0], :]
        v2 = shape[:, face_id[:, 1], :]
        v3 = shape[:, face_id[:, 2], :]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = e1.cross(e2)  # compute normal for each face , 每一個三角形網格的normal
        empty = torch.zeros((face_norm.size(0), 1, 3), dtype=face_norm.dtype, device=face_norm.device)
        face_norm = torch.cat((face_norm, empty), 1)  # concat face_normal with a zero vector at the end
        v_norm = face_norm[:, point_id, :].sum(2)  # compute vertex normal using one-ring neighborhood  #應是用鄰居來輔助計算norm
        v_norm = v_norm / v_norm.norm(dim=2).unsqueeze(2)  # normalize normal vectors

        return v_norm

    # project 3D face onto image plane
    # input: face_shape with shape [1,N,3]
    # output: face_projection with shape [1,N,2]
    # p.s 與原本的code不同，rotation & translation 在這之前就做完了
    def Projection_block(self, face_shape):  # we choose the focal length and camera position empirically
        half_image_width = self.img_size // 2
        batchsize = face_shape.shape[0]
        camera_pos = torch.tensor([0.0, 0.0, 10.0], device=face_shape.device).reshape(1, 1, 3)
        # tensor.reshape(constant([0.0,0.0,10.0]),[1,1,3])
        p_matrix = np.array([self.focal, 0.0, half_image_width, \
                             0.0, self.focal, half_image_width, \
                             0.0, 0.0, 1.0], dtype=np.float32)

        p_matrix = np.tile(p_matrix.reshape(1, 3, 3), [batchsize, 1, 1])
        reverse_z = np.tile(np.reshape(np.array([1.0, 0, 0, 0, 1, 0, 0, 0, -1.0], dtype=np.float32), [1, 3, 3]),
                            [batchsize, 1, 1])

        p_matrix = torch.tensor(p_matrix, device=face_shape.device)
        reverse_z = torch.tensor(reverse_z, device=face_shape.device)
        face_shape = torch.matmul(face_shape, reverse_z) + camera_pos
        aug_projection = torch.matmul(face_shape, p_matrix.permute((0, 2, 1)))

        face_projection = aug_projection[:, :, :2] / \
                          torch.reshape(aug_projection[:, :, 2], [batchsize, -1, 1])
        return face_projection

    # compute rotation matrix based on 3 Euler angles
    # input: angles with shape [1,3]
    # output: rotation matrix with shape [1,3,3]
    @staticmethod
    def Compute_rotation_matrix(angles):
        n_b = angles.size(0)
        sinx = torch.sin(angles[:, 0])
        siny = torch.sin(angles[:, 1])
        sinz = torch.sin(angles[:, 2])
        cosx = torch.cos(angles[:, 0])
        cosy = torch.cos(angles[:, 1])
        cosz = torch.cos(angles[:, 2])

        rotXYZ = torch.eye(3).view(1, 3, 3).repeat(n_b * 3, 1, 1).view(3, n_b, 3, 3)

        if angles.is_cuda: rotXYZ = rotXYZ.cuda()

        rotXYZ[0, :, 1, 1] = cosx
        rotXYZ[0, :, 1, 2] = -sinx
        rotXYZ[0, :, 2, 1] = sinx
        rotXYZ[0, :, 2, 2] = cosx
        rotXYZ[1, :, 0, 0] = cosy
        rotXYZ[1, :, 0, 2] = siny
        rotXYZ[1, :, 2, 0] = -siny
        rotXYZ[1, :, 2, 2] = cosy
        rotXYZ[2, :, 0, 0] = cosz
        rotXYZ[2, :, 0, 1] = -sinz
        rotXYZ[2, :, 1, 0] = sinz
        rotXYZ[2, :, 1, 1] = cosz

        rotation = rotXYZ[2].bmm(rotXYZ[1]).bmm(rotXYZ[0])

        return rotation.permute(0, 2, 1)

    @staticmethod
    def Rigid_transform_block(face_shape, rotation, translation):
        face_shape_r = torch.matmul(face_shape, rotation)
        face_shape_t = face_shape_r + translation.view(-1, 1, 3)

        return face_shape_t

    # compute vertex color using face_texture and SH function lighting approximation
    # input: face_texture with shape [1,N,3]
    #        norm with shape [1,N,3]
    #        gamma with shape [1,27]
    # Accord. to other paper , gamma is composed of light position, ambient color and diffuse color
    # output: face_color with shape [1,N,3], RGB order, range from 0-255
    #         lighting with shape [1,N,3], color under uniform texture (這份沒多輸出這個)
    @staticmethod
    def Illumination_layer(face_texture, norm, gamma):
        n_b, num_vertex, _ = face_texture.size()
        n_v_full = n_b * num_vertex
        gamma = gamma.view(-1, 3, 9).clone()
        gamma[:, :, 0] += 0.8

        gamma = gamma.permute(0, 2, 1)

        a0 = np.pi
        a1 = 2 * np.pi / np.sqrt(3.0)
        a2 = 2 * np.pi / np.sqrt(8.0)
        c0 = 1 / np.sqrt(4 * np.pi)
        c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
        c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
        d0 = 0.5 / np.sqrt(3.0)

        Y0 = torch.ones(n_v_full).to(gamma.device).float() * a0 * c0
        norm = norm.view(-1, 3)
        nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
        arrH = []

        arrH.append(Y0)
        arrH.append(-a1 * c1 * ny)
        arrH.append(a1 * c1 * nz)
        arrH.append(-a1 * c1 * nx)
        arrH.append(a2 * c2 * nx * ny)
        arrH.append(-a2 * c2 * ny * nz)
        arrH.append(a2 * c2 * d0 * (3 * nz.pow(2) - 1))
        arrH.append(-a2 * c2 * nx * nz)
        arrH.append(a2 * c2 * 0.5 * (nx.pow(2) - ny.pow(2)))

        H = torch.stack(arrH, 1)
        Y = H.view(n_b, num_vertex, 9)
        lighting = Y.bmm(gamma)

        face_color = face_texture * lighting
        return face_color

    # 應該是 compute 68 landmarks(keypoints) on image plane
    @staticmethod
    def get_lms(face_shape, kp_inds):
        lms = face_shape[:, kp_inds, :]
        return lms

    def forward(self, x):
        coeff = x['coeff']
        batch_num = coeff.shape[0]

        id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = self.Split_coeff(coeff)

        face_shape = self.Shape_formation(id_coeff, ex_coeff)
        # print("Face Shape : ",face_shape.size())
        face_texture = self.Texture_formation(tex_coeff)
        face_norm = self.Compute_norm(face_shape)
        # print(face_norm.size())
        rotation = self.Compute_rotation_matrix(angles)
        face_norm_r = face_norm.bmm(rotation)
        face_shape_t = self.Rigid_transform_block(face_shape, rotation, translation)
        # print(face_texture.size(), face_norm_r.size(), gamma.size())
        face_texture = self.Illumination_layer(face_texture, face_norm_r, gamma)
        face_lms_t = self.get_lms(face_shape_t, self.kp_inds)
        lms = self.Projection_block(face_lms_t)
        lms = torch.stack([lms[:, :, 0], self.img_size - lms[:, :, 1]], dim=2)

        face_color = TexturesVertex(face_texture)

        tri = self.tri - 1
        mesh = Meshes(verts=face_shape_t, faces=tri.repeat(batch_num, 1, 1),
                      textures=face_color)  # Meshes(Vertex, Faces, Texture)
        rendered_img = self.renderer(mesh)  # 應是設定好了從canonical view來project
        rendered_img = torch.clamp(rendered_img, 0, 255)

        return rendered_img, lms, face_texture, mesh, coeff, tri

class Pure_3DMM_UV(Pure_3DMM):

    def forward(self, x):
        coeff = x['coeff']
        uvmap = x['uvmap']
        uv_displace_map = x['uv_displace_map']
        uv_coord_2 = x['uv_coord_2']
        need_illu = True
        if x.get('need_illu') is not None:
            need_illu = x['need_illu']
        batch_num = coeff.shape[0]
        id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = self.Split_coeff(coeff)
        face_shape_ori = self.Shape_formation(id_coeff, ex_coeff) #(b, 35709, 3)

        # Apply displacement on the shape texture
        uv_vertex = F.grid_sample(uv_displace_map, uv_coord_2.float(), mode='bilinear')
        uv_vertex = torch.squeeze(uv_vertex, 2)
        uv_vertex = uv_vertex.permute(0,2,1)
        uv_vertex*= torch.tensor(0.01).to(self.device)
        face_shape = face_shape_ori + uv_vertex

        # Reconstruct face geometry from shape texture
        face_norm = self.Compute_norm(face_shape)

        rotation = self.Compute_rotation_matrix(angles)
        face_norm_r = face_norm.bmm(rotation)
        face_shape_t = self.Rigid_transform_block(face_shape, rotation, translation)
        #print(face_texture.size(), face_norm_r.size(), gamma.size())
        #face_color = self.Illumination_layer(face_texture, face_norm_r, gamma)
        face_lms_t = self.get_lms(face_shape_t, self.kp_inds)
        lms = self.Projection_block(face_lms_t)
        lms = torch.stack([lms[:, :, 0], self.img_size-lms[:, :, 1]], dim=2)
        tri = self.tri - 1

        face_texture = F.grid_sample(uvmap.permute(0,3,1,2), uv_coord_2.float(), mode='bilinear')
        #print(torch.sum(face_texture!=2))
        face_texture = torch.squeeze(face_texture, 2)
        face_texture = face_texture.permute(0,2,1)
        if(need_illu==True):
            face_color = self.Illumination_layer(face_texture, face_norm_r, gamma)
            face_color = TexturesVertex(face_color)
        else:
            face_color = TexturesVertex(face_texture)
        #print(face_texture.size())
        #print(face_color._faces_uvs_list)
        mesh = Meshes(verts=face_shape_t, faces=tri.repeat(batch_num, 1, 1), textures=face_color) # Meshes(Vertex, Faces, Texture)
        #print(mesh.verts_normals_list()[0])
        rendered_img = self.renderer(mesh) # 應是設定好了從canonical view來project
        rendered_img = torch.clamp(rendered_img, 0, 255)
        return rendered_img, lms, face_shape, mesh, coeff, tri



