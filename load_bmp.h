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
 * Bitmaps .bmp loader
 * 
 * File header
 *  
 */

#ifndef _LOAD_BMP_H
#define _LOAD_BMP_H


 
/**********************************************************
 *
 * VARIABLES DECLARATION
 *
 *********************************************************/

extern int num_texture;// Counter to keep track of the last loaded texture



/**********************************************************
 *
 * FUNCTIONS DECLARATION
 *
 *********************************************************/

extern int LoadBMP(char *p_filename);

#endif