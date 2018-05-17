#include "GLUTCallBack.h"

int main(int argc, char** argv)
{
	Scene scene;
	return glutmain(argc, argv, 320, 320, "SCV Path Tracing Demo", &scene);
}
