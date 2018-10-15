//
//  jhGuideLine.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 15..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

class jhGuideLine : CALayer {
    
    var layer_size : CGSize = CGSize.init(width: 0, height: 0)
    var lineX : CGFloat
    var lineY : CGFloat
    var lineWidth : CGFloat
    
    init?(x: CGFloat, y: CGFloat, lineWidth: CGFloat, layer:Any) {
        if GS.shared.logLevel.contains(.graph2) { print("jhGuideLine_init") }
        self.lineX = x
        self.lineY = y
        self.lineWidth = lineWidth
        super.init(layer: layer)
        //        super.isGeometryFlipped = true
        super.backgroundColor = UIColor.white.cgColor
        super.isHidden = false
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func draw(in ctx: CGContext) {
        
        if GS.shared.logLevel.contains(.graph2) { print("jhGuideLine_draw, layer_size.width = \(layer_size.width), layer_size.height = \(layer_size.height), lineX = \(lineX), lineY = \(lineY)") }
        
        layer_size = self.bounds.size
        ctx.move(to : CGPoint(x : lineX, y : 0))
        ctx.addLine(to: CGPoint(x: lineX, y: UIScreen.main.bounds.height)) //TODO: will be changed to current Scene Size
        ctx.setLineWidth(lineWidth)
        ctx.setStrokeColor(UIColor(red: 0, green: 185, blue: 255, alpha: 1.0).cgColor)
        ctx.strokePath()
        
    }
    
}
