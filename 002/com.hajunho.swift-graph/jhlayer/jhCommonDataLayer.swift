//
//  jhLayer.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 19..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

class jhCommonDataLayer<T> : CALayer {
    
    internal var panelID: Int
    internal var mValuesOfDatas : Array<CGFloat> = Array()
    var xDistance : CGFloat
    internal var superScene: T?
    
    init(_ x: jhPanel<T>, _ layer: Any) {
        
        self.xDistance = x.xDistance
        self.panelID = x.jhPanelID
        self.superScene = x.superScene
        super.init(layer: layer)
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    func drawPoint(_ ctx: CGContext, _ x : CGFloat, _ y : CGFloat, _ width : CGFloat, _ height : CGFloat, thickness : CGFloat, _ color : CGColor){
    }
}
