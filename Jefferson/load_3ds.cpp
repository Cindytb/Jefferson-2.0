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
* 3ds models loader
*
*/

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <io.h>
#include "object.h"
#include "load_3ds.h"



/**********************************************************
*
* FUNCTION Load3DS(obj_type_ptr p_object, char *p_filename)
*
* This function loads a mesh from a 3ds file.
* Please note that we are loading only the vertices, polygons and mapping lists.
* If you need to load meshes with advanced features as for example:
* multi objects, materials, lights and so on, you must insert other chunk parsers.
*
* Input parameters: p_object = Pointer to the Object
*					 p_filename = Filename of the object to load
*
* Return value: (char) 1 if the object was loaded correctly, 0 otherwise
*
*********************************************************/

char Load3DS(obj_type_ptr p_object, const char *p_filename)
{
	int i; //Index variable

	FILE *l_file; //File pointer

	unsigned short l_chunk_id; //Chunk identifier
	unsigned int l_chunk_length; //Chunk length

	unsigned char l_char; //Char variable
	unsigned short l_qty; //Number of elements in each chunk

	unsigned short l_face_flags; //Flag that stores some face information

	if (p_filename == '\0') return(0);

	if ((l_file = fopen(p_filename, "rb")) == NULL)
	{
		MessageBox(NULL, "3DS file not found", "Spacesim", MB_OK | MB_ICONERROR);
		return (0);
	}

	while (ftell(l_file) < filelength(fileno(l_file))) //Loop to scan the whole file 
	{
		//getche(); //Insert this command for debug (to wait for keypress for each chuck reading)

		fread(&l_chunk_id, 2, 1, l_file); //Read the chunk header
		//printf("ChunkID: %x\n", l_chunk_id);
		fread(&l_chunk_length, 4, 1, l_file); //Read the length of the chunk
		//printf("ChunkLength: %x\n", l_chunk_length);

		switch (l_chunk_id)
		{
			//----------------- MAIN3DS -----------------
			// Description: Main chunk, contains all the other chunks
			// Chunk ID: 4d4d 
			// Chunk Length: 0 + sub chunks
			//-------------------------------------------
		case 0x4d4d:
			break;

			//----------------- EDIT3DS -----------------
			// Description: 3D Editor chunk, objects layout info 
			// Chunk ID: 3d3d (hex)
			// Chunk Length: 0 + sub chunks
			//-------------------------------------------
		case 0x3d3d:
			break;

			//--------------- EDIT_OBJECT ---------------
			// Description: Object block, info for each object
			// Chunk ID: 4000 (hex)
			// Chunk Length: len(object name) + sub chunks
			//-------------------------------------------
		case 0x4000:
			i = 0;
			do
			{
				fread(&l_char, 1, 1, l_file);
				p_object->name[i] = l_char;
				i++;
			} while (l_char != '\0' && i<20);
			break;

			//--------------- OBJ_TRIMESH ---------------
			// Description: Triangular mesh, contains chunks for 3d mesh info
			// Chunk ID: 4100 (hex)
			// Chunk Length: 0 + sub chunks
			//-------------------------------------------
		case 0x4100:
			break;

			//--------------- TRI_VERTEXL ---------------
			// Description: Vertices list
			// Chunk ID: 4110 (hex)
			// Chunk Length: 1 x unsigned short (number of vertices) 
			//             + 3 x float (vertex coordinates) x (number of vertices)
			//             + sub chunks
			//-------------------------------------------
		case 0x4110:
			fread(&l_qty, sizeof(unsigned short), 1, l_file);
			p_object->vertices_qty = l_qty;
			//printf("Number of vertices: %d\n", l_qty);
			if (l_qty>MAX_VERTICES)
			{
				MessageBox(NULL, "Number of vertices too high", "Spacesim", MB_OK | MB_ICONERROR);
				return (0);
			}
			for (i = 0; i<l_qty; i++)
			{
				fread(&p_object->vertex[i].x, sizeof(float), 1, l_file);
				//printf("Vertices list x: %f\n", p_object->vertex[i].x);
				fread(&p_object->vertex[i].y, sizeof(float), 1, l_file);
				//printf("Vertices list y: %f\n", p_object->vertex[i].y);
				fread(&p_object->vertex[i].z, sizeof(float), 1, l_file);
				//printf("Vertices list z: %f\n", p_object->vertex[i].z);
			}
			break;

			//--------------- TRI_FACEL1 ----------------
			// Description: Polygons (faces) list
			// Chunk ID: 4120 (hex)
			// Chunk Length: 1 x unsigned short (number of polygons) 
			//             + 3 x unsigned short (polygon points) x (number of polygons)
			//             + sub chunks
			//-------------------------------------------
		case 0x4120:
			fread(&l_qty, sizeof(unsigned short), 1, l_file);
			p_object->polygons_qty = l_qty;
			//printf("Number of polygons: %d\n", l_qty);
			if (l_qty>MAX_POLYGONS)
			{
				MessageBox(NULL, "Number of polygons too high", "Spacesim", MB_OK | MB_ICONERROR);
				return (0);
			}
			for (i = 0; i<l_qty; i++)
			{
				fread(&p_object->polygon[i].a, sizeof(unsigned short), 1, l_file);
				//printf("Polygon point a: %d\n", p_object->polygon[i].a);
				fread(&p_object->polygon[i].b, sizeof(unsigned short), 1, l_file);
				//printf("Polygon point b: %d\n", p_object->polygon[i].b);
				fread(&p_object->polygon[i].c, sizeof(unsigned short), 1, l_file);
				//printf("Polygon point c: %d\n", p_object->polygon[i].c);
				fread(&l_face_flags, sizeof(unsigned short), 1, l_file);
				//printf("Face flags: %x\n", l_face_flags);
			}
			break;

			//------------- TRI_MAPPINGCOORS ------------
			// Description: Vertices list
			// Chunk ID: 4140 (hex)
			// Chunk Length: 1 x unsigned short (number of mapping points) 
			//             + 2 x float (mapping coordinates) x (number of mapping points)
			//             + sub chunks
			//-------------------------------------------
		case 0x4140:
			fread(&l_qty, sizeof(unsigned short), 1, l_file);
			for (i = 0; i<l_qty; i++)
			{
				fread(&p_object->mapcoord[i].u, sizeof(float), 1, l_file);
				//printf("Mapping list u: %f\n", p_object->mapcoord[i].u);
				fread(&p_object->mapcoord[i].v, sizeof(float), 1, l_file);
				//printf("Mapping list v: %f\n", p_object->mapcoord[i].v);
			}
			break;

			//----------- Skip unknow chunks ------------
			//We need to skip all the chunks that currently we don't use
			//We use the chunk length information to set the file pointer
			//to the same level next chunk
			//-------------------------------------------
		default:
			fseek(l_file, l_chunk_length - 6, SEEK_CUR);
		}
	}
	fclose(l_file); // Closes the file stream
	return (1); // Returns ok
}
