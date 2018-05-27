#ifndef IMGUIGLUT_H
#define IMGUIGLUT_H

#include <imgui.h>

IMGUI_API bool        ImGui_ImplGLUT_Init();
IMGUI_API void        ImGui_ImplGLUT_NewFrame(int w, int h);
IMGUI_API void        ImGui_ImplGLUT_Shutdown();

#endif