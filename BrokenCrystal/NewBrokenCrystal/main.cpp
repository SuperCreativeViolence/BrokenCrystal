#include "GLUTCallBack.h"

int main(int argc, char** argv)
{
	// 384, 216
	return glutmain(argc, argv, 1280, 720, "SCV Path Tracing Demo", Scene::GetInstance());
}
