#include "GLUTCallBack.h"

int main(int argc, char** argv)
{
	Scene scene;
	return glutmain(argc, argv, 384, 216, "SCV Path Tracing Demo", &scene);
}
