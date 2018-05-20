#include "GLUTCallBack.h"

int main(int argc, char** argv)
{
	Scene scene;
	return glutmain(argc, argv, 1280, 720, "SCV Path Tracing Demo", &scene);
}
