#include "DebugDrawer.h"
#include "Scene.h"

void DebugDrawer::drawLine(const btVector3 &from, const btVector3 &to, const btVector3 &color)
{
	glBegin(GL_LINES);
	glColor3f(color.getX(), color.getY(), color.getZ());
	glVertex3f(from.getX(), from.getY(), from.getZ());
	glVertex3f(to.getX(), to.getY(), to.getZ());
	glEnd();
}

void DebugDrawer::setDebugMode(int debugMode)
{
	debugMode = debugMode;
}

int DebugDrawer::getDebugMode() const
{
	return debugMode;
}

void DebugDrawer::drawContactPoint(const btVector3 &pointOnB, const btVector3 &normalOnB, btScalar distance, int lifeTime, const btVector3 &color)
{
	btVector3 const startPoint = pointOnB;
	btVector3 const endPoint = pointOnB + normalOnB * distance;
	drawLine(startPoint, endPoint, color);
}

void DebugDrawer::ToggleDebugFlag(int flag)
{
	if (debugMode & flag)
		debugMode = debugMode & (~flag);
	else
		debugMode |= flag;
}