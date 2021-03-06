#include "graphics.cuh"
#undef HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
////////////////////////////////////////////////////////////////////////////////

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;


double tan40 = tan(40.0 * PI / 180);
StopWatchInterface *timer = NULL;

float g_fAnim = 0.0;

//ball variables
//float coordinates.x = 0.5, coordinates.y = 0.0, coordinates.z = 0.0;
float ball_rotate_x = 0.0, ball_rotate_y = 0.0, ball_rotate_z = 0.0;
float temp = 0.05f;
//float temp = 0.1f;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;

float translate_z = -3.0;
float translate_x = 0;

// Absolute rotation values (0-359 degrees) and rotation increments for each frame
double rotation_x = 0, rotation_x_increment = 0.1;
double rotation_y = 0, rotation_y_increment = 0.05;
double rotation_z = 0, rotation_z_increment = 0.03;

// Flag for rendering as lines or filled polygons
int filling = 1; //0=OFF 1=ON

//Lights settings
GLfloat light_ambient[] = { 0.1f, 0.1f, 0.1f, 1.0f };
GLfloat light_diffuse[] = { 0.2f, 0.2f, 0.2f, 1.0f };
GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat light_position[] = { 0.0f, 1.0f, 1.0f, 1.0f };

//Materials settings
GLfloat mat_ambient[] = { 0.1f, 0.1f, 0.1f, 0.0f };
GLfloat mat_diffuse[] = { 0.2f, 0.2f, 0.2f, 0.0f };
GLfloat mat_specular[] = { 0.2f, 0.2f, 0.2f, 0.0f };
GLfloat mat_shininess[] = { 0.01f };

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;
const char *sSDKsample = "Cindy Bui Final Project";
Data *GP;
std::stringstream s;


VBO *obj;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int graphicsMain(int argc, char **argv, Data *p)
{
	char *ref_file = NULL;
	GP = p;
	pArgc = &argc;
	pArgv = argv;

#if defined(__linux__)
	setenv("DISPLAY", ":0", 0);
#endif

	printf("%s starting...\n", sSDKsample);

	if (argc > 1)
	{
		if (checkCmdLineFlag(argc, (const char **)argv, "file"))
		{
			// In this mode, we are running non-OpenGL and doing a compare of the VBO was generated correctly
			getCmdLineArgumentString(argc, (const char **)argv, "file", (char **)&ref_file);
		}
	}

	printf("\n");

	runTest(argc, argv, ref_file);

	printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
	//exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
	return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv, char *ref_file)
{

	// Create the CUTIL timer
	sdkCreateTimer(&timer);
	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	if (false == initGL(&argc, argv))
	{
		return false;
	}
	// register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutSpecialFunc(specialKeys);
#if defined (__APPLE__) || defined(MACOSX)
	atexit(cleanup);
#else
	glutCloseFunc(cleanup);
#endif
	
#if(DEBUGMODE != 1)
	/*MOVING SIGNAL TO GPU*/
	// Allocate device memory for signal
	//SoundSource* source = &(GP->all_sources[0]);
	//float *d_signal;
	//checkCudaErrors(cudaMalloc((void **)&d_signal, source->length * sizeof(float)));

	//// Copy signal from host to device
	//checkCudaErrors(cudaMemcpy(d_signal, source->buf, source->length * sizeof(float),
	//	cudaMemcpyHostToDevice));
	//source->waveform = new VBO(&d_signal, &translate_x, source->length, 1 / 44100.0f);
	//source->waveform->init();
	//source->waveform->averageNum = 100;
	//source->waveform->create();
#endif
	// create sine wave VBO
	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

	// run the cuda part
	runCuda(&cuda_vbo_resource);

	// start rendering mainloop
	glutMainLoop();

	return true;
}

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////


void launch_kernel(float4 *pos, unsigned int mesh_width,
	unsigned int mesh_height, float time)
{
	// execute the kernel
	dim3 block(8, 8, 1);
	dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	simple_vbo_kernel << < grid, block >> > (pos, mesh_width, mesh_height, time);
}


void computeFPS()
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		fpsCount = 0;
		fpsLimit = (int)MAX(avgFPS, 1.f);

		sdkResetTimer(&timer);
	}

	char fps[256];
	sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
	glutSetWindowTitle(fps);
}

void loadModel() {
	aiLogStream stream;
	/* get a handle to the predefined STDOUT log stream and attach
   it to the logging system. It remains active for all further
   calls to aiImportFile(Ex) and aiApplyPostProcessing. */
	stream = aiGetPredefinedLogStream(aiDefaultLogStream_STDOUT, NULL);
	aiAttachLogStream(&stream);

	/* ... same procedure, but this stream now writes the
	   log messages to assimp_log.txt */
	stream = aiGetPredefinedLogStream(aiDefaultLogStream_FILE, "assimp_log.txt");
	aiAttachLogStream(&stream);

	/*Load the model*/
	if (0 != loadasset("media/Jefferson_Colored.fbx")) {
		printf("ERROR LOADING FBX FILE\n");
		exit(EXIT_FAILURE);
		
	}
}
////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
	/*Old*/
	glutInit(argc, argv);
	//glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Cindy Bui Final Project");
	glutDisplayFunc(display);
	glutIdleFunc(display);

	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	// initialize necessary OpenGL extensions
	if (!isGLVersionSupported(2, 0))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// default initialization
	//RGB: 193 215 229
	float red = 193.0f / 256.0f;
	float green = 215.0f / 256.0f;
	float blue = 229.0f / 256.0f;
	glClearColor(red, green, blue, 1.0);

	// viewport
	glViewport(0, 0, window_width, window_height);

	//Lights initialization and activation
	glLightfv(GL_LIGHT1, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT1, GL_SPECULAR, light_specular);
	glLightfv(GL_LIGHT1, GL_POSITION, light_position);
	glEnable(GL_LIGHTING);
	//glEnable(GL_LIGHT0);
	glEnable(GL_LIGHT1);

	//Other initializations
	glShadeModel(GL_SMOOTH); // Type of shading for the polygons
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST); // Texture mapping perspective correction (OpenGL... thank you so much!)
	glEnable(GL_TEXTURE_2D); // Texture mapping ON
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); // Polygon rasterization mode (polygon filled)
	//glEnable(GL_CULL_FACE); // Enable the back face culling

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 10.0);

	loadModel();
	// glewInit();
	SDK_CHECK_ERROR_GL();

	return true;
}


////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
	// map OpenGL buffer object for writing from CUDA
	float4 *dptr;
	checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
		*vbo_resource));

	launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

#ifdef _WIN32
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#endif


////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
	unsigned int vbo_res_flags)
{
	assert(vbo);

	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

	//SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}
void moveBar(Data p) {
	if (p.pauseStatus == true) {
		return;
	}
	SoundSource* curr_source = (SoundSource*)&(GP->all_sources[0]);
	translate_x = (float)curr_source->count * -(curr_source->waveform->ratio);
}
////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
	sdkStartTimer(&timer);

	// run CUDA kernel to generate vertex positions
	//runCuda(&cuda_vbo_resource);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);

	glLoadIdentity();
	/*zooming in & out*/
	glTranslatef(0.0, 0.0, translate_z);

	/*Rotating around the mesh's axis*/
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);
	/*Step up the phase for the sinusoidal waves*/
	g_fAnim += 0.01f;
	
	/*Setup animation for waveform VBO*/
	//moveBar(*GP);
#if(DEBUGMODE != 1)
	SoundSource* source = (SoundSource*) &(GP->all_sources[0]);
	source->updateFromCartesian();
	/*source->azi += 3.0f;
	if (source->azi > 360) {
		source->azi -= 360;
	}
	source->updateFromSpherical();*/
	
	//source->drawWaveform();
#endif
	
	/////////////////////////////////////////////////////////
	/*Render the floor mesh*/
	/////////////////////////////////////////////////////////

	glPushMatrix();
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f( 47.0f/256.0f, 63.0f/256.0f, 45.0f/256.0f );
	glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
	glDisableClientState(GL_VERTEX_ARRAY);
	glPopMatrix();


	/////////////////////////////////////////////////////////
	/*Render Jefferson*/
	/////////////////////////////////////////////////////////
	glPushMatrix();
	float tmp = scene_max.x - scene_min.x;
	tmp = aisgl_max(scene_max.y - scene_min.y, tmp);
	tmp = aisgl_max(scene_max.z - scene_min.z, tmp);
	tmp = 1.f / tmp;

	/*Move Jefferson down so his ears are at the the origin*/
	glTranslatef(-0.05f, -0.2f, 0.00f);
	/*Scale Jefferson to a proper size*/
	glScalef(tmp, tmp, tmp);
	/* center the model */
	glTranslatef(-scene_center.x, -scene_center.y, -scene_center.z);
	

	/* if the display list has not been made yet, create a new one and
			fill it with scene contents */
	if (scene_list == 0) {
		scene_list = glGenLists(1);
		glNewList(scene_list, GL_COMPILE);
		/* now begin at the root node of the imported data and traverse
			   the scenegraph by multiplying subsequent local transforms
			   together on GL's matrix stack. */
		recursive_render(scene, scene->mRootNode);
		glEndList();
	}
	glCallList(scene_list);
	glPopMatrix();

	/*Sound source sphere*/
	glPushMatrix();
	glEnable(GL_DEPTH_TEST);
	glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_COLOR_MATERIAL);
	GLUquadricObj *quadric;
	quadric = gluNewQuadric();
	gluQuadricTexture(quadric, GL_TRUE);
	gluQuadricNormals(quadric, GLU_SMOOTH);
	//Source RGB: 119, 207, 131
	float red = 119.0f / 256.0f;
	float green = 207.0f / 256.0f;
	float blue = 131.0f / 256.0f;
	SoundSource* curr_source = (SoundSource*)&(p->all_sources[0]);
	glColor3f(red, green, blue);
	glTranslatef(curr_source->coordinates.x, curr_source->coordinates.y, curr_source->coordinates.z);
	gluSphere(quadric, 0.1, 20, 50);
	glPopMatrix();

	//curr_source->updateFromCartesian();
	/*printf("Cartesian: %.3f %.3f %.3f\n", p->all_sources[0].coordinates.x, p->all_sources[0].coordinates.y, p->all_sources[0].coordinates.z);
	printf("Spherical: %3f %3f %3f\n", p->all_sources[0].azi, p->all_sources[0].ele, p->all_sources[0].r);*/
	/*Push out the OpenGL buffer*/ 
	glutSwapBuffers();
	sdkStopTimer(&timer);
	computeFPS();
}
void timerEvent(int value){
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}
void cleanup()
{	
	printf("Cleaning up\n");
	#if(DEBUGMODE != 1)
		closeEverything();
	#endif
	sdkDeleteTimer(&timer);
	
	if (vbo)
	{
		deleteVBO(&vbo, cuda_vbo_resource);
	}
	cudaDeviceReset();
}
////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	SoundSource* source = (SoundSource*) &(GP->all_sources[0]);
	float dist = std::sqrt(source->coordinates.x * source->coordinates.x + source->coordinates.z * source->coordinates.z);
	/*Calculate the radius, distance, elevation, and azimuth*/
	//float ele = (float)atan2(coordinates.y, dist) * 180.0f / PI;
	switch (key)
	{
	case('r'):
		rotate_x = 0.0f;
		rotate_y = 0.0f;
		translate_z = -3.0f;
		source->coordinates.x = 0.5;
		source->coordinates.y = 0.0;
		source->coordinates.z = 0.0;
		break;
	case('W'):
	case('w'):
		//value is 40 degrees in radians
		if (source->coordinates.y >= 0 || source->coordinates.y < 0 && (atan((source->coordinates.y + temp) / dist) * 180.0f / PI > -40))
			source->coordinates.y += temp;
		break;
	case('S'):
	case('s'):
		if (source->coordinates.y >= 0 || source->coordinates.y < 0 && (atan((source->coordinates.y - temp) / dist) * 180.0f / PI > -40))
			source->coordinates.y -= temp;

		break;
		/*TODO: Fix this logic*/
	case('A'):
	case('a'):
		if (atan(source->coordinates.y / std::sqrt((source->coordinates.x - temp) * (source->coordinates.x - temp) + source->coordinates.z * source->coordinates.z)) * 180.0f / PI > -40)
			source->coordinates.x -= temp;
		break;
	case('D'):
	case('d'):
		if (atan(source->coordinates.y / std::sqrt((source->coordinates.x + temp) * (source->coordinates.x + temp) + source->coordinates.z * source->coordinates.z)) * 180.0f / PI > -40)
			source->coordinates.x += temp;
		break;
	case (27):
		printf("Finished playout\n");
#if defined(__APPLE__) || defined(MACOSX)
		exit(EXIT_SUCCESS);
#else
		glutDestroyWindow(glutGetWindow());
		return;
#endif
	}
}
void specialKeys(int key, int x, int y) {
	SoundSource* source = (SoundSource*) &(GP->all_sources[0]);
	switch (key) {
	case GLUT_KEY_LEFT:
		if (atan(source->coordinates.y / std::sqrt((source->coordinates.x - temp) * (source->coordinates.x - temp) + source->coordinates.z * source->coordinates.z)) * 180.0f / PI > -40)
			source->coordinates.x -= temp;
		break;
	case GLUT_KEY_RIGHT:
		if (atan(source->coordinates.y / std::sqrt((source->coordinates.x + temp) * (source->coordinates.x + temp) + source->coordinates.z * source->coordinates.z)) * 180.0f / PI > -40)
			source->coordinates.x += temp;
		break;
	case GLUT_KEY_UP:
		if (atan(source->coordinates.y / std::sqrt(source->coordinates.x * source->coordinates.x + (source->coordinates.z - temp) * (source->coordinates.z - temp))) * 180.0f / PI > -40)
			source->coordinates.z -= temp;
		break;
	case GLUT_KEY_DOWN:
		if (atan(source->coordinates.y / std::sqrt(source->coordinates.x * source->coordinates.x + (source->coordinates.z + temp) * (source->coordinates.z + temp))) * 180.0f / PI > -40)
			source->coordinates.z += temp;
		break;
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
		if (button == 3)
		{
			translate_z += 0.1f;
		}
		else if (button == 4)
		{
			translate_z -= 0.1f;
		}
		//printf("Scroll %s At %d %d\n", (button == 3) ? "Up" : "Down", x, y);
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}
	mouse_old_x = x;
	mouse_old_y = y;
}
/*Rotate the perspective*/
void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4)
	{
		translate_z += dy * 0.01f;
	}
	mouse_old_x = x;
	mouse_old_y = y;
}