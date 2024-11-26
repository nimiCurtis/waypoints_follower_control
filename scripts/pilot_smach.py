#!/usr/bin/env python3
import rospy
import smach
import threading
from geometry_msgs.msg import PoseStamped
from pilot_utils.utils import clip_angles
from tf.transformations import quaternion_from_euler
from pilot_goal_generation_publisher import GoalGenerator

import numpy as np

# Define State for using model prediction
class ModelPredictionState(smach.State):
    def __init__(self, goal_generator: GoalGenerator):
        smach.State.__init__(self, outcomes=['to_goal_zone', 'to_goal_reached'])
        self.goal_generator = goal_generator

    def execute(self, userdata):
        rospy.loginfo("Using model predictions...")
        self.goal_generator.transformed_pose = self.goal_generator.model_predict()
        if self.goal_generator.goal_reached.data:
            return 'to_goal_reached'
        elif self.goal_generator.observed_target and self.goal_generator.is_goal_in_zone():
            return 'to_goal_zone'
        return 'to_goal_zone'


# Define State for when robot is in the goal zone
class GoalZoneState(smach.State):
    def __init__(self, goal_generator: GoalGenerator):
        smach.State.__init__(self, outcomes=['to_model_prediction', 'to_goal_reached'])
        self.goal_generator = goal_generator

    def execute(self, userdata):
        rospy.loginfo("Robot in goal zone, fine-tuning position...")
        if abs(self.goal_generator.dgoal) <= 0.7 and self.goal_generator.observed_target:
            desired_pose = self.goal_generator.calculate_goal_zone_pose()
            self.goal_generator.transformed_pose = desired_pose
            self.goal_generator.goal_pub_sensor.publish(self.goal_generator.transformed_pose)
            if self.goal_generator.goal_reached.data:
                return 'to_goal_reached'
        return 'to_model_prediction'


# Define State for when goal is reached
class GoalReachedState(smach.State):
    def __init__(self, goal_generator: GoalGenerator):
        smach.State.__init__(self, outcomes=['to_model_prediction'])
        self.goal_generator = goal_generator

    def execute(self, userdata):
        rospy.loginfo("Goal reached! Publishing final pose...")
        pose_in_base = PoseStamped()
        yaw = clip_angles(np.arctan2(self.goal_generator.goal_to_target[1], self.goal_to_target[0]))
        q = quaternion_from_euler(0, 0, yaw)
        pose_in_base.pose.orientation.x = q[0]
        pose_in_base.pose.orientation.y = q[1]
        pose_in_base.pose.orientation.z = q[2]
        pose_in_base.pose.orientation.w = q[3]
        pose_in_base.header.frame_id = self.goal_generator.base_frame
        pose_in_base.header.stamp = rospy.Time.now()
        self.goal_generator.goal_pub_sensor.publish(pose_in_base)
        return 'to_model_prediction'


# Helper to check if goal is in zone
def is_goal_in_zone(goal_generator):
    return abs(goal_generator.dgoal) <= 0.7


if __name__ == '__main__':
    rospy.init_node('goal_state_machine')

    # Instantiate the GoalGenerator
    goal_generator = GoalGenerator()

    # Start data collection in a background thread
    threading.Thread(target=data_collection_thread, args=(goal_generator,), daemon=True).start()

    # Create the state machine
    sm = smach.StateMachine(outcomes=['succeeded', 'aborted', 'preempted'])

    with sm:
        smach.StateMachine.add('MODEL_PREDICTION', ModelPredictionState(goal_generator),
                               transitions={
                                   'to_goal_zone': 'GOAL_ZONE',
                                   'to_goal_reached': 'GOAL_REACHED'
                               })

        smach.StateMachine.add('GOAL_ZONE', GoalZoneState(goal_generator),
                               transitions={
                                   'to_model_prediction': 'MODEL_PREDICTION',
                                   'to_goal_reached': 'GOAL_REACHED'
                               })

        smach.StateMachine.add('GOAL_REACHED', GoalReachedState(goal_generator),
                               transitions={
                                   'to_model_prediction': 'MODEL_PREDICTION'
                               })

    # Execute the state machine
    outcome = sm.execute()