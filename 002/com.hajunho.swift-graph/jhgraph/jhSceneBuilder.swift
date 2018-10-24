//
//  jhSceneBuilder.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 24..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

enum sceneType {
    case NORMAL
    case TIMELINE
}

class jhSceneBuilder {
    
    private var x: CGFloat
    private var y: CGFloat
    private var width: CGFloat
    private var height: CGFloat
    private var mType: sceneType
    
    init() {
        mType = .NORMAL
        x = 0
        y = 0
        width = UIScreen.main.bounds.width
        height = UIScreen.main.bounds.height
    }
    
    @discardableResult
    func type(_ x: sceneType) -> jhSceneBuilder {
        self.mType = x
        return self
    }
    
    @discardableResult
    func frame(_ x: CGFloat, _ y: CGFloat, _ width: CGFloat, _ height: CGFloat) -> jhSceneBuilder {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        return self
    }
    
    @discardableResult
    func build() -> jhScene {
        switch mType {
        case .TIMELINE:
            var ret = jhSceneTimeLine(frame: CGRect(x: x, y: y, width: width, height: height))
            temp(ret: &ret)
            return ret
            
        case .NORMAL:
            var ret = jhScene(frame: CGRect(x: x, y: y, width: width, height: height))
            temp(ret: &ret)
            return ret
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
