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
 * File header
 *  
 */

#ifndef _OBJECT_H
#define _OBJECT_H

#include "mat_vect.h"
#include "mat_matr.h"

#define MAX_OBJECTS 100 // Max number of objects



/**********************************************************
 *
 * TYPES DECLARATION
 *
 *********************************************************/

#define MAX_VERTICES 100000 // Max number of vertices (for each object)
#define MAX_POLYGONS 100000 // Max number of polygons (for each object)

// Our vertex type
typedef struct{
    float x,y,z;
}vertex_type;

// The polygon (triangle), 3 numbers that aim 3 vertices
typedef struct{
    unsigned short a,b,c;
}polygon_type;

// The mapcoord type, 2 texture coordinates for each vertex
typedef struct{
    float u,v;
}mapcoord_type;

// The object type
typedef struct {

	char name[20]; // Name of the object
    
	int vertices_qty; // Number of vertices
    int polygons_qty; // Number of polygons

    vertex_type vertex[MAX_VERTICES]; // Array of vertices
    vertex_type normal[MAX_VERTICES]; // Array of the vertices' normals
    polygon_type polygon[MAX_POLYGONS]; // Array of polygons (numbers that point to the vertices' list)
    mapcoord_type mapcoord[MAX_VERTICES]; // Array of U,V coordinates for texture mapping

    int id_texture; // Number of the texture 

	matrix_4x4_type matrix; // Object matrix

} obj_type, *obj_type_ptr;



/**********************************************************
 *
 * VARIABLES DECLARATION
 *
 *********************************************************/

extern obj_type object[MAX_OBJECTS];
extern int obj_qty;
extern int obj_control;



/**********************************************************
 *
 * FUNCTIONS DECLARATION
 *
 *********************************************************/

extern char ObjLoad(char *p_object_name, char *p_texture_name, float p_pos_x, float p_pos_y, float p_pos_z, int p_rot_x, int p_rot_y, int p_rot_z);
extern char ObjLoad(const char *p_object_name);
extern void ObjCalcNormals(obj_type_ptr p_object);
extern void ObjPosition (obj_type_ptr p_object,float p_x,float p_y,float p_z);
extern void ObjTranslate (obj_type_ptr p_object,float p_x,float p_y,float p_z);
extern void ObjTranslateW (obj_type_ptr p_object,float p_x,float p_y,float p_z);
extern void ObjRotate (obj_type_ptr p_object,int p_angle_x,int p_angle_y,int p_angle_z);

#endif