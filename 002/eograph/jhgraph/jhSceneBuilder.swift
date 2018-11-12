//
//  jhSceneBuilder.swift
//  bridge8
//
//  Created by Junho HA on 2022. 2. 22.
//  Copyright © 2022년 eoflow. All rights reserved.
//

import UIKit

public enum sceneType {
    case NORMAL
    case TIMELINE
}

open class jhSceneBuilder<T : jhScene> {
    
    private var x: CGFloat
    private var y: CGFloat
    private var width: CGFloat
    private var height: CGFloat
    public var mType: sceneType
    
    public init() {
        mType = .NORMAL
        x = 0
        y = 0
        width = UIScreen.main.bounds.width
        height = UIScreen.main.bounds.height
    }
    
    @discardableResult
    public func type(_ x: sceneType) -> jhSceneBuilder {
        self.mType = x
        return self
    }
    
    @discardableResult
    public func frame(_ x: CGFloat, _ y: CGFloat, _ width: CGFloat, _ height: CGFloat) -> jhSceneBuilder {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        return self
    }
    
    @discardableResult
    public func build<T>() -> T {
        switch mType {
        case .TIMELINE:
            var ret = jhSceneTimeLine(frame: CGRect(x: x, y: y, width: width, height: height))
            temp(ret: &ret)
            return ret as! T
            
        case .NORMAL:
            var ret = jhScene(frame: CGRect(x: x, y: y, width: width, height: height))
            temp(ret: &ret)
            return ret as! T
        }
    }
    
    private func temp<T: jhScene>(ret: inout T) {
        ret.contentSize = CGSize(width: ret.frame.width*4, height: ret.frame.height+100) //TODO:
        ret.isUserInteractionEnabled = true
        ret.translatesAutoresizingMaskIntoConstraints = true
        ret.maximumZoomScale = 4.0
        ret.minimumZoomScale = 0.1
        ret.isScrollEnabled = true
        ret.backgroundColor = UIColor.white
    }
}
