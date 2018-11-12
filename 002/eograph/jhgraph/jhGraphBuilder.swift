//
//  jhGraphBuilder.swift
//  bridge8
//
//  Created by Junho HA on 2022. 2. 22
//  Copyright © 2022년 eoflow. All rights reserved.
//

import UIKit

public enum graphType {
    case LINE
    case BAR
    case TYPE1
    case TYPE2
    case TYPE3
    case TYPE4
}




open class jhGraphBuilder<T> {
    
  
    
    private var x : CGFloat
    private var y : CGFloat
    private var width : CGFloat
    private var height : CGFloat
    
    private var mGtype : graphType
    private var superScene : T?
    
    init() {
        mGtype = graphType.LINE
        x = 0
        y = 0
        width = UIScreen.main.bounds.width
        height = UIScreen.main.bounds.height
        superScene = nil
    }
    
    @discardableResult
    func scene(_ x: T) -> jhGraphBuilder<T> {
        self.superScene = x
        if jhGS.s.logLevel.contains(.network2) {
            print("ctime in jhGraphBuilder_scene = ", (x as? jhSceneTimeLine)?.currentTime)
            print("ctime in jhGraphBuilder_scene2 = ", (self.superScene as? jhSceneTimeLine)?.currentTime )
        }
        return self
    }
    
    @discardableResult
    func type(_ x: graphType) -> jhGraphBuilder<T> {
        self.mGtype = x
        return self
    }
    
    @discardableResult
    func frame(_ x: CGFloat, _ y: CGFloat, _ width: CGFloat, _ height: CGFloat) -> jhGraphBuilder<T> {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        return self
    }
    
    @discardableResult
    func build() -> jhPanel<T> {
        if jhGS.s.logLevel.contains(.network2) {
            print("ctime in jhGraphBuilder_build = ", (self.superScene as? jhSceneTimeLine)?.currentTime)
        }
        switch mGtype {
        case .LINE:
            return jhLineGraph<T>(frame: CGRect(x: x, y: y, width: width, height: height))
        case .BAR:
            return jhBarGraph<T>(frame: CGRect(x: x, y: y, width: width, height: height))
        case .TYPE1:
            return jhType1graphPanel<T>(frame: CGRect(x: x, y: y, width: width, height: height), scene: &superScene)
        case .TYPE4:
            return jhType4graphPanel<T>(frame: CGRect(x: x, y: y, width: width, height: height), scene: &superScene)
        case .TYPE2:
            return jhType2graphPanel<T>(frame: CGRect(x: x, y: y, width: width, height: height), scene: &superScene)
        case .TYPE3:
            return jhType3graphPanel<T>(frame: CGRect(x: x, y: y, width: width, height: height), scene: &superScene)
        }
    }
}

