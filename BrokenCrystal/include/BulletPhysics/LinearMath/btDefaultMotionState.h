#ifndef BT_DEFAULT_MOTION_STATE_H
#define BT_DEFAULT_MOTION_STATE_H

#include "btMotionState.h"

///The btDefaultMotionState provides a common implementation to synchronize world transforms with offsets.
ATTRIBUTE_ALIGNED16(struct)	btDefaultMotionState : public btMotionState
{
	btTransform m_graphicsWorldTrans;
	btTransform	m_centerOfMassOffset;
	btTransform m_startWorldTrans;
	void*		m_userPointer;

	BT_DECLARE_ALIGNED_ALLOCATOR();

	btDefaultMotionState(const btTransform& startTrans = btTransform::getIdentity(),const btTransform& centerOfMassOffset = btTransform::getIdentity())
		: m_graphicsWorldTrans(startTrans),
		m_centerOfMassOffset(centerOfMassOffset),
		m_startWorldTrans(startTrans),
		m_userPointer(0)

	{
	}

	///synchronizes world transform from user to physics
	virtual void	getWorldTransform(btTransform& centerOfMassWorldTrans ) const 
	{
			centerOfMassWorldTrans = m_graphicsWorldTrans * m_centerOfMassOffset.inverse() ;
	}

	///synchronizes world transform from physics to user
	///Bullet only calls the update of worldtransform for active objects
	virtual void	setWorldTransform(const btTransform& centerOfMassWorldTrans)
	{
			m_graphicsWorldTrans = centerOfMassWorldTrans * m_centerOfMassOffset;
	}

	virtual btVector3   GetWorldPosition()
	{
		return (m_graphicsWorldTrans * m_centerOfMassOffset.inverse()).getOrigin();
	}

	virtual btQuaternion GetWorldRotation()
	{
		return (m_graphicsWorldTrans * m_centerOfMassOffset.inverse()).getRotation();
		btMatrix3x3 matrix = btMatrix3x3((m_graphicsWorldTrans * m_centerOfMassOffset.inverse()).getRotation());

	}

};

#endif //BT_DEFAULT_MOTION_STATE_H
