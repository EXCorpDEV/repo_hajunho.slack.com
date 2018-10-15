//
//  jhGraphBuilder.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 15..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

class jhGraphBuilder {
    
    private var x : CGFloat
    private var y : CGFloat
    private var width : CGFloat
    private var height : CGFloat
    private var mGtype : graphType
    
    init() {
        mGtype = graphType.LINE
        x = 0
        y = 0
        width = UIScreen.main.bounds.width
        height = UIScreen.main.bounds.width
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
    func build() -> jhPanel {
        switch mGtype {
            
        case .LINE:
            return jhLineGraph(frame: CGRect(x: x, y: y, width: width, height: height))
        case .BAR:
            return jhBarGraph(frame: CGRect(x: x, y: y, width: width, height: height))
        }
    }
    
}
