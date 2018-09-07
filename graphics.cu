#include "graphics.cuh"

////////////////////////////////////////////////////////////////////////////////

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;


double tan40 = tan(40.0 * PI / 180);
StopWatchInterface *timer = NULL;

float g_fAnim = 0.0;

//ball variables
float ball_x = 0.5, ball_y = 0.0, ball_z = 0.0;
float ball_rotate_x = 0.0, ball_rotate_y = 0.0, ball_rotate_z = 0.0;
float temp = 0.005f;

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
GLfloat light_ambient[] = { 0.3f, 0.3f, 0.3f, 0.3f };
GLfloat light_diffuse[] = { 0.2f, 0.2f, 0.2f, 0.2f };
GLfloat light_specular[] = { 0.2f, 0.2f, 0.2f, 0.2f };
GLfloat light_position[] = { 0.0f, 50.0f, 1.0f, 1.0f };

//Materials settings
GLfloat mat_ambient[] = { 0.1f, 0.1f, 0.1f, 0.0f };
GLfloat mat_diffuse[] = { -0.2f, -0.2f, -0.2f, -0.0f };
GLfloat mat_specular[] = { -0.2f, -0.2f, -0.2f, -0.0f };
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
PaStream *stream;
PaError err;


VBO **obj;

int numObj;
__global__ void myNewKernel(int id, float4 *pos, float *d_buf, unsigned int size, double ratio, int magicNumber) {
	/*My version of the kernel*/
	unsigned long modNum = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long samp_num = modNum + (id * magicNumber);
	if (modNum < size && samp_num > id * magicNumber && samp_num < (id + 1) * magicNumber) {
		float x = (float)samp_num * ratio;
		float y = d_buf[samp_num];
		/*Flat 2D waveform for testing*/
		pos[modNum] = make_float4(x, y, 0, 1.0f);
	}
}

void launch_new_kernel(int id, float4 *pos, float *buf, unsigned int size, int magicNumber) {
	unsigned const int numThreads = 1024;
	int numBlocks;
	if (size == magicNumber) {
		numBlocks = 1280;
	}
	else {
		numBlocks = size / numThreads + 1;
	}
	myNewKernel << < numBlocks, numThreads >> > (id, pos, buf, size, ratio, magicNumber);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	fprintf(stderr, "Kernel launch successful\n\n");
	// unmap buffer object
}

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// calculate uv coordinates
	float u = x / (float)width;
	float v = y / (float)height;
	u = u * 2.0f - 1.0f;
	v = v * 2.0f - 1.0f;

	// calculate simple sine wave pattern
	float freq = 4.0f;
	float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

	// write output vertex
	pos[y*width + x] = make_float4(u, w, v, 1.0f);
}

void launch_kernel(float4 *pos, unsigned int mesh_width,
	unsigned int mesh_height, float time)
{
	// execute the kernel
	dim3 block(8, 8, 1);
	dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	simple_vbo_kernel << < grid, block >> > (pos, mesh_width, mesh_height, time);
}

bool checkHW(char *name, const char *gpuType, int dev)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	strcpy(name, deviceProp.name);

	if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType)))
	{
		return true;
	}
	else
	{
		return false;
	}
}

int findGraphicsGPU(char *name)
{
	int nGraphicsGPU = 0;
	int deviceCount = 0;
	bool bFoundGraphics = false;
	char firstGraphicsName[256], temp[256];

	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess)
	{
		printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("> FAILED %s sample finished, exiting...\n", sSDKsample);
		exit(EXIT_FAILURE);
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
	{
		printf("> There are no device(s) supporting CUDA\n");
		return false;
	}
	else
	{
		printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
	}

	for (int dev = 0; dev < deviceCount; ++dev)
	{
		bool bGraphics = !checkHW(temp, (const char *)"Tesla", dev);
		printf("> %s\t\tGPU %d: %s\n", (bGraphics ? "Graphics" : "Compute"), dev, temp);

		if (bGraphics)
		{
			if (!bFoundGraphics)
			{
				strcpy(firstGraphicsName, temp);
			}

			nGraphicsGPU++;
		}
	}

	if (nGraphicsGPU)
	{
		strcpy(name, firstGraphicsName);
	}
	else
	{
		strcpy(name, "this hardware");
	}

	return nGraphicsGPU;
}
void initVboArray(int magicNumber) {
	for (int i = 0; i < numObj; i++) {
		obj[i]->init();
		obj[i]->create(magicNumber);
	}
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int graphicsMain(int argc, char **argv, Data *p, PaError err2, PaStream *stream2)
{
	char *ref_file = NULL;
	GP = p;
	err = err2;
	stream = stream2;
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
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHT1);
	glEnable(GL_LIGHT2);
	glEnable(GL_LIGHT3);

	//Materials initialization and activation
	glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);

	glEnable(GL_COLOR_MATERIAL);
	//Other initializations
	glShadeModel(GL_SMOOTH); // Type of shading for the polygons
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST); // Texture mapping perspective correction (OpenGL... thank you so much!)
	glEnable(GL_TEXTURE_2D); // Texture mapping ON
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); // Polygon rasterization mode (polygon filled)
	glEnable(GL_CULL_FACE); // Enable the back face culling

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)window_width / (GLfloat)window_height, 0.1, 10.0);

	/*Attempting to create a face*/
	printf("...Loading body\n");
	ObjLoad("body.3ds");
	printf("...Loading eyes\n");
	ObjLoad("eyes.3ds");
	printf("...Loading smile\n");
	ObjLoad("smile.3ds");
	printf("...Loading letter\n");
	ObjLoad("letter.3ds");
	printf("...Loading hat\n");
	ObjLoad("hat.3ds");
	glewInit();
	SDK_CHECK_ERROR_GL();

	return true;
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
	const unsigned int magicNumber = 1310720;
	if (GP->length <= magicNumber) {
		numObj = 1;
	}
	else {
		numObj = GP->length / magicNumber + 1;
	}
#if(DEBUGMODE != 1)
	/*MOVING SIGNAL TO GPU*/
	// Allocate device memory for signal
	float *d_signal;
	checkCudaErrors(cudaMalloc((void **)&d_signal, GP->length * sizeof(float)));

	// Copy signal from host to device
	checkCudaErrors(cudaMemcpy(d_signal, GP->buf, GP->length * sizeof(float),
		cudaMemcpyHostToDevice));

	obj = new VBO *[numObj];
	for (int i = 0; i < numObj; i++) {
		obj[i] = new VBO(&d_signal, &translate_x);
	}
	for (int i = 0; i < numObj; i++) {
		obj[i]->id = i;
		if (i == numObj - 1) {
			obj[i]->numSamples = GP->length - i * magicNumber;
		}
		else {
			obj[i]->numSamples = magicNumber;
		}
	}
	initVboArray(magicNumber);
#endif
	// create sine wave VBO
	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

	// run the cuda part
	runCuda(&cuda_vbo_resource);

	// start rendering mainloop
	glutMainLoop();

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

void sdkDumpBin2(void *data, unsigned int bytes, const char *filename)
{
	printf("sdkDumpBin: <%s>\n", filename);
	FILE *fp;
	FOPEN(fp, filename, "wb");
	fwrite(data, bytes, 1, fp);
	fflush(fp);
	fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runAutoTest(int devID, char **argv, char *ref_file)
{
	char *reference_file = NULL;
	void *imageData = malloc(mesh_width*mesh_height * sizeof(float));

	// execute the kernel
	launch_kernel((float4 *)d_vbo_buffer, mesh_width, mesh_height, g_fAnim);

	cudaDeviceSynchronize();
	getLastCudaError("launch_kernel failed");

	checkCudaErrors(cudaMemcpy(imageData, d_vbo_buffer, mesh_width*mesh_height * sizeof(float), cudaMemcpyDeviceToHost));

	sdkDumpBin2(imageData, mesh_width*mesh_height * sizeof(float), "simpleGL.bin");
	reference_file = sdkFindFilePath(ref_file, argv[0]);

	if (reference_file &&
		!sdkCompareBin2BinFloat("simpleGL.bin", reference_file,
			mesh_width*mesh_height * sizeof(float),
			MAX_EPSILON_ERROR, THRESHOLD, pArgv[0]))
	{
		g_TotalErrors++;
	}
}

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
	translate_x = (float)p.count * -ratio;
}
////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
	sdkStartTimer(&timer);

	// run CUDA kernel to generate vertex positions
	runCuda(&cuda_vbo_resource);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);

	glLoadIdentity();
	/*zooming in & out*/
	glTranslatef(0.0, 0.0, translate_z);

	/*Rotating around the mesh's axis*/
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);
	moveBar(*GP);
#if(DEBUGMODE != 1)
	/*Calculate the radius, distance, elevation, and azimuth*/
	float r = std::sqrt(ball_x * ball_x + ball_z * ball_z + ball_y * ball_y);
	float horizR = std::sqrt(ball_x * ball_x + ball_z * ball_z);
	float ele = (float)atan2(ball_y, horizR) * 180.0f / PI;
	//s.str(std::string());
	float obj_azi = (float)atan2(ball_x / r, ball_z / r) * 180.0f / PI;
	/*s << "Azimuth: " << obj_azi;
	s << "Elevation: " << ele;
	s << "Radius: " << r;*/
	GP->hrtf_idx = pick_hrtf(ele, obj_azi);
	float newR = r / 100 + 1;
	GP->gain = 1 / pow(newR, 2);
	
	float rotateVBO_y = (float)atan2(-ball_z, ball_x) * 180.0f / PI;

	if (rotateVBO_y < 0) {
		rotateVBO_y += 360;
	}
	//printf("x: %3f\ty: %3f\tz: %3f\tX: %3f\tY: %3f\tZ: %3f\n", ball_x, ball_y, ball_z, rotateVBO_x, rotateVBO_y, rotateVBO_z);
#endif
	for (int i = 0; i < numObj; i++) {
		obj[i]->draw(rotateVBO_y, ele);
	}

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);

	/*SINE WAVE COLORS*/
	glColor3f( 47.0f/256.0f, 63.0f/256.0f, 45.0f/256.0f );
	glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
	glDisableClientState(GL_VERTEX_ARRAY);

	
	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	GLUquadricObj *quadric;
	quadric = gluNewQuadric();
	gluQuadricTexture(quadric, GL_TRUE);
	gluQuadricNormals(quadric, GLU_SMOOTH);

	/*IT'S JEFFERSON*/
	for (int i = 0; i<obj_qty; i++)
	{
		//glTranslatef(0.0, 0.0, 0.0);
		if(i == 0) glColor3f(60.0f / 256.0f, 52.0f / 256.0f, 96.0f / 256.0f);
		else if (i == 1) glColor3f(0.0f, 1.0f, 1.0f);
		else if (i == 4) glColor3f(0.0f, 0.0f, 0.0f);
		else glColor3f(1.0f, 1.0f, 1.0f);
		glPushMatrix(); // We save the current matrix
		glScalef(0.5f, 0.5f, 0.5f);
		//glTranslatef(ball_x, ball_y, ball_z);
		glTranslatef(0.0, 0.0, 0.0);
		glMultMatrixf(&object[i].matrix[0][0]); // Now let's multiply the object matrix by the identity-first matrix

		if (object[i].id_texture != -1)
		{
			glBindTexture(GL_TEXTURE_2D, object[i].id_texture); // We set the active texture 
			glEnable(GL_TEXTURE_2D); // Texture mapping ON
		}
		else
			glDisable(GL_TEXTURE_2D); // Texture mapping OFF

		glBegin(GL_TRIANGLES); // glBegin and glEnd delimit the vertices that define a primitive (in our case triangles)
		for (int j = 0; j<object[i].polygons_qty; j++)
		{
			//----------------- FIRST VERTEX -----------------
			//Normal coordinates of the first vertex
			glNormal3f(object[i].normal[object[i].polygon[j].a].x,
				object[i].normal[object[i].polygon[j].a].y,
				object[i].normal[object[i].polygon[j].a].z);
			// Texture coordinates of the first vertex
			glTexCoord2f(object[i].mapcoord[object[i].polygon[j].a].u,
				object[i].mapcoord[object[i].polygon[j].a].v);
			// Coordinates of the first vertex
			glVertex3f(object[i].vertex[object[i].polygon[j].a].x,
				object[i].vertex[object[i].polygon[j].a].y,
				object[i].vertex[object[i].polygon[j].a].z);

			//----------------- SECOND VERTEX -----------------
			//Normal coordinates of the second vertex
			glNormal3f(object[i].normal[object[i].polygon[j].b].x,
				object[i].normal[object[i].polygon[j].b].y,
				object[i].normal[object[i].polygon[j].b].z);
			// Texture coordinates of the second vertex
			glTexCoord2f(object[i].mapcoord[object[i].polygon[j].b].u,
				object[i].mapcoord[object[i].polygon[j].b].v);
			// Coordinates of the second vertex
			glVertex3f(object[i].vertex[object[i].polygon[j].b].x,
				object[i].vertex[object[i].polygon[j].b].y,
				object[i].vertex[object[i].polygon[j].b].z);

			//----------------- THIRD VERTEX -----------------
			//Normal coordinates of the third vertex
			glNormal3f(object[i].normal[object[i].polygon[j].c].x,
				object[i].normal[object[i].polygon[j].c].y,
				object[i].normal[object[i].polygon[j].c].z);
			// Texture coordinates of the third vertex
			glTexCoord2f(object[i].mapcoord[object[i].polygon[j].c].u,
				object[i].mapcoord[object[i].polygon[j].c].v);
			// Coordinates of the Third vertex
			glVertex3f(object[i].vertex[object[i].polygon[j].c].x,
				object[i].vertex[object[i].polygon[j].c].y,
				object[i].vertex[object[i].polygon[j].c].z);

		}
		glEnd();
		glPopMatrix(); // Restore the previous matrix 
	}

	/*SOUND SOURCE SPHERE*/
	//Source RGB: 119, 207, 131
	float red = 119.0f / 256.0f;
	float green = 207.0f / 256.0f;
	float blue = 131.0f / 256.0f;
	glColor3f(red, green, blue);
	glPushMatrix();
	glTranslatef(ball_x, ball_y, ball_z);
	gluSphere(quadric, 0.1, 20, 50);
	glPopMatrix();


	/*GL Setup to display text onto the screen for debugging purposes*/
	//glMatrixMode(GL_PROJECTION);
	//glPushMatrix();
	//glLoadIdentity();
	//glMatrixMode(GL_MODELVIEW);
	//glPushMatrix();
	//glLoadIdentity();
	//glDisable(GL_DEPTH_TEST);

	//glColor3f(255, 255, 255);
	//glRasterPos2f(0,0);
	//std::string temp = s.str();
	//int len = (int) temp.length();
	//for (int i = 0; i < len; i++) {
	//	glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, temp.at(i));
	//}
	//glEnable(GL_DEPTH_TEST); // Turn depth testing back on

	//glMatrixMode(GL_PROJECTION);
	//glPopMatrix(); // revert back to the matrix I had before.
	//glMatrixMode(GL_MODELVIEW);
	//glPopMatrix();

	/*Step up the phase for the sinusoidal waves*/
	g_fAnim += 0.01f;

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
	sdkDeleteTimer(&timer);

	if (vbo)
	{
		deleteVBO(&vbo, cuda_vbo_resource);
	}
#if(DEBUGMODE != 1)
	/*Close output file*/
	sf_close(GP->sndfile);

	/* Stop stream */
	err = Pa_StopStream(stream);
	if (err != paNoError) {
		printf("PortAudio error: stop stream: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		exit(1);
	}

	/* Close stream */
	err = Pa_CloseStream(stream);
	if (err != paNoError) {
		printf("PortAudio error: close stream: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		exit(1);
	}

	/* Terminate PortAudio */
	err = Pa_Terminate();
	if (err != paNoError) {
		printf("PortAudio error: terminate: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		exit(1);
	}

#endif
}
////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	float dist = std::sqrt(ball_x * ball_x + ball_z * ball_z);

	switch (key)
	{
	case('r'):
		rotate_x = 0.0f;
		rotate_y = 0.0f;
		translate_z = -3.0f;
		ball_x = 0.5;
		ball_y = 0.0;
		ball_z = 0.0;
		break;
	case('w'):
		//value is 40 degrees in radians
		if (ball_y >= 0 || ball_y < 0 && (atan(ball_y / dist) < 0.6981317))
			ball_y += temp;
		break;
	case('s'):
		ball_y -= temp;
		break;
		/*TODO: Fix this logic*/
	case('a'):
		if (ball_y >= 0 || ball_y < 0 && (atan(ball_y / std::sqrt(pow(ball_x - temp, 2) + pow(ball_z, 2)) < tan40)))
			ball_x -= temp;
		break;
	case('d'):
		if (ball_y >= 0 || ball_y < 0 && (atan(ball_y / std::sqrt(pow(ball_x + temp, 2) + pow(ball_z, 2)) < tan40)))
			ball_x += temp;
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

	switch (key) {
	case GLUT_KEY_LEFT:
		if (ball_y <= 0 || ball_y > 0 && (atan(ball_y / std::sqrt(pow(ball_x - temp, 2) + pow(ball_z, 2)) < tan40)))
			ball_x -= temp;
		break;
	case GLUT_KEY_RIGHT:
		if (ball_y <= 0 || ball_y > 0 && (atan(ball_y / std::sqrt(pow(ball_x + temp, 2) + pow(ball_z, 2)) < tan40)))
			ball_x += temp;
		break;
	case GLUT_KEY_UP:
		if (ball_y <= 0 || ball_y > 0 && (atan(ball_y / std::sqrt(pow(ball_x, 2) + pow(ball_z - temp, 2)) < tan40)))
			ball_z -= temp;
		break;
	case GLUT_KEY_DOWN:
		if (ball_y <= 0 || ball_y > 0 && (atan(ball_y / std::sqrt(pow(ball_x, 2) + pow(ball_z + temp, 2)) < tan40)))
			ball_z += temp;
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


VBO::VBO(float **a, float *b)
	: d_buf(a), translate_x(b)
{
}
void VBO::init() {
	assert(&vbo);

	// create buffer object
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	vboSize = numSamples * sizeof(float4);
	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, vboSize, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard));
	printf("ID: %i, VBO: %i\n", id, vbo);
	fprintf(stderr, "ID: %i, VBO: %i\n", id, vbo);
	SDK_CHECK_ERROR_GL();
}
void VBO::create(int magicNumber) {
	// map OpenGL buffer object for writing from CUDA
	float4 *dptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
		cuda_vbo_resource));
	printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

	fprintf(stderr, "\n\n\nLaunching Kernel: id: %d, number of samples: %d\n", id, numSamples);
	launch_new_kernel(id, dptr, *d_buf, numSamples, magicNumber);

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}
void VBO::update(
void VBO::draw(float rotateVBO_y, float rotateVBO_z) {
	glPushMatrix();
	
	// render from the vbo
	glRotatef(rotateVBO_y, 0, 1.0, 0);
	glRotatef(rotateVBO_z, 0, 0, 1.0);
	glTranslatef(*translate_x, 0.0, 0.0);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 1.0, 1.0);
	glDrawArrays(GL_POINTS, 0, vboSize);
	glDisableClientState(GL_VERTEX_ARRAY);

	glPopMatrix();
}


VBO::~VBO() {
	checkCudaErrors(cudaFree(*d_buf));
	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

	glBindBuffer(1, vbo);
	glDeleteBuffers(1, &vbo);

	vbo = 0;
}