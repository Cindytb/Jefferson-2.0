/*
 * ---------------- www.spacesimulator.net --------------
 *   ---- Space simulators and 3d engine tutorials ----
 *
 * Author: Damiano Vitulli
 *
 * This program is released under the BSD licence
 * By using this program you agree to licence terms on spacesimulator.net copyright page
 *
 *
 * Object functions
 *  
 */

#include <math.h>
#include <stdio.h>
#include "load_3ds.h"
#include "load_bmp.h"
#include "object.h"



/**********************************************************
 *
 * VARIABLES DECLARATION
 *
 *********************************************************/

obj_type object[MAX_OBJECTS]; //Now the object is generic, the cube has annoyed us a little bit, or not?
int obj_qty=0; //Number of objects in our world
int obj_control=0; //Number of the object that we can control



/**********************************************************
 *
 * FUNCTION ObjLoad(char *p_object_name, char *p_texture_name, float p_pos_x, float p_pos_y, float p_pos_z, float p_rot_x, float p_rot_y, float p_rot_z)
 *
 * This function loads an object and set some object's data
 *
 * Parameters: p_object_name = object name
 *			   p_texture_name = texture name
 *             p_pos_x = starting position x coordinate
 *             p_pos_y = starting position y coordinate
 *             p_pos_z = starting position z coordinate
 *             p_rot_x = starting orientation x axis
 *             p_rot_y = starting orientation y axis
 *             p_rot_z = starting orientation z axis
 *
 * Return value: (char) 1 if the object was loaded correctly, 0 otherwise
 *
 *********************************************************/

char ObjLoad(char *p_object_name, char *p_texture_name, float p_pos_x, float p_pos_y, float p_pos_z, int p_rot_x, int p_rot_y, int p_rot_z)
{
    if (Load3DS (&object[obj_qty],p_object_name)==0) return(0); //Object loading
   // object[obj_qty].id_texture=LoadBMP(p_texture_name); // The Function LoadBitmap() returns the current texture ID
	ObjCalcNormals(&object[obj_qty]); //Once we have all the object data we need to calculate all the normals of the object's vertices
	MatrIdentity_4x4(object[obj_qty].matrix); //Object matrix init
	ObjPosition(&object[obj_qty], p_pos_x, p_pos_y, p_pos_z); // Object initial position
	ObjRotate(&object[obj_qty], p_rot_x, p_rot_y, p_rot_z); // Object initial rotation
	obj_qty++; // Let's increase the object number and get ready to load another object!
	return (1); // If all is ok then return 1
}

char ObjLoad(const char *p_object_name) {
	float p_pos_x = 0.0f;
	float p_pos_y = 0.0f;
	float p_pos_z = 0.0f;
	float p_rot_x = 0.0f;
	float p_rot_y = 0.0f;
	float p_rot_z = 0.0f;
	printf("Loading %s\n", p_object_name);
	if (Load3DS(&object[obj_qty], p_object_name) == 0) return(0); //Object loading
	ObjCalcNormals(&object[obj_qty]); //Once we have all the object data we need to calculate all the normals of the object's vertices
	MatrIdentity_4x4(object[obj_qty].matrix); //Object matrix init
	ObjPosition(&object[obj_qty], p_pos_x, p_pos_y, p_pos_z); // Object initial position
	ObjRotate(&object[obj_qty], p_rot_x, p_rot_y, p_rot_z); // Object initial rotation
	obj_qty++; // Let's increase the object number and get ready to load another object!
	return (1); // If all is ok then return 1
}

/**********************************************************
 *
 * SUBROUTINE ObjCalcNormals(obj_type_ptr p_object)
 *
 * This function calculate all the polygons and vertices' normals of the specified object
 *
 * Input parameters: p_object = object
 *
 *********************************************************/

void ObjCalcNormals(obj_type_ptr p_object)
{
	int i;
	p3d_type l_vect1,l_vect2,l_vect3,l_vect_b1,l_vect_b2,l_normal;  //Some local vectors
	int l_connections_qty[MAX_VERTICES]; //Number of poligons around each vertex

    // Resetting vertices' normals...
	for (i=0; i<p_object->vertices_qty; i++)
	{
		p_object->normal[i].x = 0.0;
		p_object->normal[i].y = 0.0;
		p_object->normal[i].z = 0.0;
		l_connections_qty[i]=0;
	}
	
	for (i=0; i<p_object->polygons_qty; i++)
	{
        l_vect1.x = p_object->vertex[p_object->polygon[i].a].x;
        l_vect1.y = p_object->vertex[p_object->polygon[i].a].y;
        l_vect1.z = p_object->vertex[p_object->polygon[i].a].z;
        l_vect2.x = p_object->vertex[p_object->polygon[i].b].x;
        l_vect2.y = p_object->vertex[p_object->polygon[i].b].y;
        l_vect2.z = p_object->vertex[p_object->polygon[i].b].z;
        l_vect3.x = p_object->vertex[p_object->polygon[i].c].x;
        l_vect3.y = p_object->vertex[p_object->polygon[i].c].y;
        l_vect3.z = p_object->vertex[p_object->polygon[i].c].z;         
  
        // Polygon normal calculation
		VectCreate (&l_vect1, &l_vect2, &l_vect_b1); // Vector from the first vertex to the second one
        VectCreate (&l_vect1, &l_vect3, &l_vect_b2); // Vector from the first vertex to the third one
        VectDotProduct (&l_vect_b1, &l_vect_b2, &l_normal); // Dot product between the two vectors
        VectNormalize (&l_normal); //Normalizing the resultant we obtain the polygon normal

		l_connections_qty[p_object->polygon[i].a]+=1; // For each vertex shared by this polygon we increase the number of connections
		l_connections_qty[p_object->polygon[i].b]+=1;
		l_connections_qty[p_object->polygon[i].c]+=1;

		p_object->normal[p_object->polygon[i].a].x+=l_normal.x; // For each vertex shared by this polygon we add the polygon normal
		p_object->normal[p_object->polygon[i].a].y+=l_normal.y;
		p_object->normal[p_object->polygon[i].a].z+=l_normal.z;
		p_object->normal[p_object->polygon[i].b].x+=l_normal.x;
		p_object->normal[p_object->polygon[i].b].y+=l_normal.y;
		p_object->normal[p_object->polygon[i].b].z+=l_normal.z;
		p_object->normal[p_object->polygon[i].c].x+=l_normal.x;
		p_object->normal[p_object->polygon[i].c].y+=l_normal.y;
		p_object->normal[p_object->polygon[i].c].z+=l_normal.z;	
	}	
	
    for (i=0; i<p_object->vertices_qty; i++)
	{
		if (l_connections_qty[i]>0)
		{
			p_object->normal[i].x /= l_connections_qty[i]; // Let's now average the polygons' normals to obtain the vertex normal!
			p_object->normal[i].y /= l_connections_qty[i];
			p_object->normal[i].z /= l_connections_qty[i];
		}
	}
}



/**********************************************************
 *
 * SUBROUTINE ObjPosition (obj_type_ptr p_object,float p_x,float p_y,float p_z)
 *
 * Object positioning relative to the world
 *
 * Input parameters: p_object = object to move
 *                   p_x = x coordinate
 *					 p_y = y coordinate
 *                   p_z = z coordinate
 *
 *********************************************************/

void ObjPosition (obj_type_ptr p_object,float p_x,float p_y,float p_z)
{
	//The position fields in the object matrix are filled with the new values
    p_object->matrix[3][0]=p_x;
    p_object->matrix[3][1]=p_y;
    p_object->matrix[3][2]=p_z;    
}



/**********************************************************
 *
 * SUBROUTINE ObjTranslate (obj_type_ptr p_object,float p_x,float p_y,float p_z)
 *
 * Object translation relative to the point of view
 *
 * Input parameters: p_object = object to translate
 *                   p_x = x coordinate
 *					 p_y = y coordinate
 *                   p_z = z coordinate
 *
 *********************************************************/

void ObjTranslate (obj_type_ptr p_object,float p_x,float p_y,float p_z)
{
    int j,k;
    matrix_4x4_type l_matrix, l_res;

    MatrIdentity_4x4(l_matrix);
    l_matrix[3][0]=p_x;
    l_matrix[3][1]=p_y;
    l_matrix[3][2]=p_z;

	//The object matrix is multiplied by a translation matrix
    MatrMul_4x4_4x4(l_matrix,p_object->matrix,l_res);
    for(j=0;j<4;j++)
      for(k=0;k<4;k++)
        p_object->matrix[j][k]=l_res[j][k];
}



/**********************************************************
 *
 * SUBROUTINE ObjTranslateW (obj_type_ptr p_object,float p_x,float p_y,float p_z)
 *
 * Object translation relative to the world
 *
 * Input parameters: p_object = object to translate
 *                   p_x = x coordinate
 *					 p_y = y coordinate
 *                   p_z = z coordinate
 *
 *********************************************************/

void ObjTranslateW (obj_type_ptr p_object,float p_x,float p_y,float p_z)
{
      p_object->matrix[3][0]+=p_x;
      p_object->matrix[3][1]+=p_y;
      p_object->matrix[3][2]+=p_z;
}



/**********************************************************
 *
 * SUBROUTINE ObjRotate (obj_type_ptr p_object,int p_angle_x,int p_angle_y,int p_angle_z)
 *
 * Object rotation relative to the point of view
 * Angles are expressed in 1/10 degree
 *
 * Input parameters: p_object = object to rotate
 *                   p_angle_x = x rotation angle
 *					 p_angle_y = y rotation angle
 *                   p_angle_z = z rotation angle
 *
 *********************************************************/

void ObjRotate (obj_type_ptr p_object,int p_angle_x,int p_angle_y,int p_angle_z)
{
    matrix_4x4_type l_matrix, l_res;

	//Range control
	if (p_angle_x<0) p_angle_x=3600+p_angle_x;
    if (p_angle_y<0) p_angle_y=3600+p_angle_y;  
    if (p_angle_z<0) p_angle_z=3600+p_angle_z;
    if (p_angle_x<0 || p_angle_x>3600) p_angle_x=0;
    if (p_angle_y<0 || p_angle_y>3600) p_angle_y=0;  
    if (p_angle_z<0 || p_angle_z>3600) p_angle_z=0;

    if (p_angle_x)
    {
		//The object matrix is multiplied by the X rotation matrix
        MatrIdentity_4x4(l_matrix);   
        l_matrix[1][1]=(matr_cos_table[p_angle_x]);
        l_matrix[1][2]=(matr_sin_table[p_angle_x]);
        l_matrix[2][1]=(-matr_sin_table[p_angle_x]);
        l_matrix[2][2]=(matr_cos_table[p_angle_x]);
        MatrMul_4x4_4x4(l_matrix,p_object->matrix,l_res);
        MatrCopy_4x4(p_object->matrix,l_res);
    }
    if (p_angle_y)
    {
		// ...by the Y rotation matrix
        MatrIdentity_4x4(l_matrix);
        l_matrix[0][0]=(matr_cos_table[p_angle_y]);
        l_matrix[0][2]=(-matr_sin_table[p_angle_y]);
        l_matrix[2][0]=(matr_sin_table[p_angle_y]);
        l_matrix[2][2]=(matr_cos_table[p_angle_y]);
        MatrMul_4x4_4x4(l_matrix,p_object->matrix,l_res);
        MatrCopy_4x4(p_object->matrix,l_res);
    }
    if (p_angle_z)
    {
		// ...by the Z rotation matrix
        MatrIdentity_4x4(l_matrix);
        l_matrix[0][0]=(matr_cos_table[p_angle_z]);
        l_matrix[0][1]=(matr_sin_table[p_angle_z]);
        l_matrix[1][0]=(-matr_sin_table[p_angle_z]);
        l_matrix[1][1]=(matr_cos_table[p_angle_z]);
        MatrMul_4x4_4x4(l_matrix,p_object->matrix,l_res);
        MatrCopy_4x4(p_object->matrix,l_res);
    }
}