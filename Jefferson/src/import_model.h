/*ASSIMP library includes*/
#include <assimp/cimport.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <GL/freeglut.h>
/* the global Assimp scene object */
extern const aiScene* scene;
extern aiVector3D scene_min, scene_max, scene_center;
extern GLuint scene_list;

#define aisgl_min(x, y) (x < y ? x : y)
#define aisgl_max(x, y) (y > x ? y : x)

/* ---------------------------------------------------------------------------- */
void get_bounding_box_for_node(const struct aiNode* nd,
	aiVector3D* min,
	aiVector3D* max,
	aiMatrix4x4* trafo);

/* ---------------------------------------------------------------------------- */
void get_bounding_box(aiVector3D* min, aiVector3D* max);

/* ---------------------------------------------------------------------------- */
void color4_to_float4(const aiColor4D* c, float f[4]);

/* ---------------------------------------------------------------------------- */
void set_float4(float f[4], float a, float b, float c, float d);

/* ---------------------------------------------------------------------------- */
void apply_material(const aiMaterial* mtl);
/* ---------------------------------------------------------------------------- */
void recursive_render(const aiScene* sc, const aiNode* nd);
/* ---------------------------------------------------------------------------- */
int loadasset(const char* path);

