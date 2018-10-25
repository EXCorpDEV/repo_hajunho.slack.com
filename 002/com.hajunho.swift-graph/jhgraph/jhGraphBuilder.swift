//
//  jhGraphBuilder.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 15..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

enum graphType {
    case LINE
    case BAR
    case TYPE1
    case TYPE4
}

struct ratioNtype {
    var ratio : CGFloat
    var type : graphType
    //    var datas : hjh
}

class jhGraphBuilder<T> {
    
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
    func scene<V>(_ x: V) -> jhGraphBuilder {
        self.superScene? = x as! T
        return self
    }
    
    @discardableResult
    func type(_ x: graphType) -> jhGraphBuilder {
        self.mGtype = x
        return self
    }
    
    @discardableResult
    func frame(_ x: CGFloat, _ y: CGFloat, _ width: CGFloat, _ height: CGFloat) -> jhGraphBuilder {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        return self
    }
    
    @discardableResult
    func build() -> jhPanel<T> {
        switch mGtype {
        case .LINE:
            return jhLineGraph<T>(frame: CGRect(x: x, y: y, width: width, height: height))
        case .BAR:
            return jhBarGraph<T>(frame: CGRect(x: x, y: y, width: width, height: height))
        case .TYPE1:
            return jhType1graphPanel<T>(frame: CGRect(x: x, y: y, width: width, height: height), scene: &superScene)
        case .TYPE4:
            return jhType4graph<T>(frame: CGRect(x: x, y: y, width: width, height: height))
        }
    }
    
}
