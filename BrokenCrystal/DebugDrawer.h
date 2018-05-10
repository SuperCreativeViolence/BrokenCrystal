#ifndef DEBUGDRAWER_H
#define DEBUGDRAWER_H

#include <BulletPhysics/LinearMath/btIDebugDraw.h>

class DebugDrawer : public btIDebugDraw
{
public:
	virtual void setDebugMode(int debugMode) override;
	virtual int getDebugMode() const override;
	virtual void drawContactPoint(const btVector3 &pointOnB, const btVector3 &normalOnB, btScalar distance, int lifeTime, const btVector3 &color) override;
	virtual void drawLine(const btVector3 &from, const btVector3 &to, const btVector3 &color) override;
	void ToggleDebugFlag(int flag);

	// 사용 안함
	virtual void reportErrorWarning(const char* warningString) override
	{
	}
	virtual void draw3dText(const btVector3 &location, const char* textString) override
	{
	}

protected:
	int debugMode;
};
#endif
